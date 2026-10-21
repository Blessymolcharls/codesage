"""Integration tests for GitHub repository workspace registration, indexing, refresh, and recovery."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codesage.github import (
    FetchResult,
    GitHubRepoIdentity,
    GithubFetchError,
    GithubIngestionError,
    InvalidGithubUrlError,
)
from codesage.retrieval import TfidfRetriever
from codesage.workspace import (
    DuplicateWorkspaceError,
    WorkspaceManager,
    WorkspaceNotFoundError,
)


@pytest.fixture
def temp_manager_env():
    """Create a temporary data directory and WorkspaceManager for GitHub tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        manager = WorkspaceManager(data_dir)
        yield {
            "manager": manager,
            "data_dir": data_dir,
            "root_tmp": Path(tmp_dir),
        }


def test_register_github_workspace(temp_manager_env):
    """Test registering a GitHub workspace establishes correct metadata and layout."""
    manager = temp_manager_env["manager"]
    url = "https://github.com/encode/uvicorn"

    ws = manager.register_github_workspace(url)
    assert ws.metadata.id == "gh-encode-uvicorn"
    assert ws.metadata.name == "encode/uvicorn"
    assert ws.metadata.source_type == "github"
    assert ws.metadata.github_url == "https://github.com/encode/uvicorn"
    assert ws.metadata.github_owner == "encode"
    assert ws.metadata.github_repo == "uvicorn"
    assert ws.metadata.status == "unindexed"
    assert ws.metadata.commit_sha is None

    # Verify directory layout
    assert ws.workspace_dir.exists()
    assert ws.cache_dir.exists()
    assert (ws.workspace_dir / "metadata.json").exists()


def test_reject_duplicate_github_registration(temp_manager_env):
    """Test registering the same GitHub repository twice is rejected unless explicitly allowed."""
    manager = temp_manager_env["manager"]
    url = "https://github.com/tiangolo/fastapi.git"

    ws1 = manager.register_github_workspace(url)
    assert ws1.id == "gh-tiangolo-fastapi"

    # Second registration with trailing slash
    with pytest.raises(DuplicateWorkspaceError, match="already registered"):
        manager.register_github_workspace("https://github.com/tiangolo/fastapi/")

    # Explicit allow_duplicate with custom ID succeeds
    ws2 = manager.register_github_workspace(
        url,
        workspace_id="fastapi-fork",
        allow_duplicate=True,
    )
    assert ws2.id == "fastapi-fork"
    assert ws2.metadata.github_url == "https://github.com/tiangolo/fastapi"


def test_workspace_id_collision_handling(temp_manager_env):
    """Test automatic suffixing when default workspace ID collides with existing workspace."""
    manager = temp_manager_env["manager"]

    # Register first repo
    ws1 = manager.register_github_workspace("https://github.com/psf/requests")
    assert ws1.id == "gh-psf-requests"

    # Register second repo that would produce the same ID if allow_duplicate is True
    ws2 = manager.register_github_workspace(
        "https://github.com/psf/requests",
        allow_duplicate=True,
    )
    assert ws2.id == "gh-psf-requests-2"
    assert ws2.workspace_dir != ws1.workspace_dir


def test_fetch_and_index_github_workspace_success(temp_manager_env):
    """Test successful clone, AST indexing, graph building, and metadata update."""
    manager = temp_manager_env["manager"]
    ws = manager.register_github_workspace("https://github.com/samuelcolvin/pydantic")

    mock_sha = "1" * 40

    def mock_fetch(identity, destination_dir, expected_parent, **kwargs):
        destination_dir.mkdir(parents=True, exist_ok=True)
        (destination_dir / "models.py").write_text(
            "class BaseModel:\n"
            "    '''Core Pydantic validation model.'''\n"
            "    def parse_obj(self, obj):\n"
            "        return obj\n",
            encoding="utf-8",
        )
        return FetchResult(
            source_dir=destination_dir,
            commit_sha=mock_sha,
            total_files=1,
            total_bytes=120,
        )

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch):
        ready_ws = manager.fetch_and_index_github_workspace(ws.id)

    assert ready_ws.metadata.status == "ready"
    assert ready_ws.metadata.commit_sha == mock_sha
    assert ready_ws.metadata.unit_count > 0
    assert ready_ws.units_path.exists()
    assert ready_ws.graph_path.exists()
    assert ready_ws.source_dir.exists()
    assert (ready_ws.source_dir / "models.py").exists()

    # Verify search on GitHub workspace
    units = ready_ws.load_code_units()
    assert any(u.get("symbol_name") == "BaseModel" for u in units)

    retriever = TfidfRetriever()
    retriever.fit(units)
    results = retriever.search("Pydantic validation model", top_k=3)
    assert len(results) > 0
    assert results[0].symbol_name == "BaseModel"


def test_refresh_unchanged_commit_returns_up_to_date(temp_manager_env):
    """Verify refresh when commit SHA is unchanged skips reindexing."""
    manager = temp_manager_env["manager"]
    ws = manager.register_github_workspace("https://github.com/pallets/click")
    mock_sha = "2" * 40

    def mock_fetch(identity, destination_dir, expected_parent, **kwargs):
        destination_dir.mkdir(parents=True, exist_ok=True)
        (destination_dir / "core.py").write_text("def command(): pass\n", encoding="utf-8")
        return FetchResult(
            source_dir=destination_dir,
            commit_sha=mock_sha,
            total_files=1,
            total_bytes=50,
        )

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch):
        ws = manager.fetch_and_index_github_workspace(ws.id)

    assert ws.metadata.commit_sha == mock_sha

    # Refresh with same SHA
    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch):
        refreshed_ws, report = manager.refresh_github_workspace(ws.id)

    assert report["status"] == "up_to_date"
    assert report["commit_sha"] == mock_sha
    assert refreshed_ws.metadata.commit_sha == mock_sha


def test_refresh_new_commit_updates_index_and_sha(temp_manager_env):
    """Verify refresh with a new commit reindexes and atomically updates commit SHA."""
    manager = temp_manager_env["manager"]
    ws = manager.register_github_workspace("https://github.com/pallets/jinja")
    sha_v1 = "a" * 40
    sha_v2 = "b" * 40

    # Initial v1
    def mock_fetch_v1(identity, destination_dir, expected_parent, **kwargs):
        destination_dir.mkdir(parents=True, exist_ok=True)
        (destination_dir / "parser.py").write_text("def parse_template(): pass\n", encoding="utf-8")
        return FetchResult(
            source_dir=destination_dir,
            commit_sha=sha_v1,
            total_files=1,
            total_bytes=50,
        )

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch_v1):
        ws = manager.fetch_and_index_github_workspace(ws.id)

    assert ws.metadata.commit_sha == sha_v1

    # Refresh with v2
    def mock_fetch_v2(identity, destination_dir, expected_parent, **kwargs):
        destination_dir.mkdir(parents=True, exist_ok=True)
        (destination_dir / "parser.py").write_text("def parse_template_v2(): pass\n", encoding="utf-8")
        (destination_dir / "lexer.py").write_text("def tokenize(): pass\n", encoding="utf-8")
        return FetchResult(
            source_dir=destination_dir,
            commit_sha=sha_v2,
            total_files=2,
            total_bytes=100,
        )

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch_v2):
        refreshed_ws, report = manager.refresh_github_workspace(ws.id)

    assert report["status"] == "updated"
    assert report["previous_commit"] == sha_v1
    assert report["new_commit"] == sha_v2
    assert refreshed_ws.metadata.commit_sha == sha_v2

    # Check updated units
    units = refreshed_ws.load_code_units()
    names = {u.get("symbol_name") for u in units}
    assert "parse_template_v2" in names
    assert "tokenize" in names


def test_failed_refresh_preserves_previous_source_and_index(temp_manager_env):
    """Verify that if refresh fails (fetch timeout or error), the previous working index is retained."""
    manager = temp_manager_env["manager"]
    ws = manager.register_github_workspace("https://github.com/pypa/pip")
    sha_v1 = "c" * 40

    def mock_fetch_ok(identity, destination_dir, expected_parent, **kwargs):
        destination_dir.mkdir(parents=True, exist_ok=True)
        (destination_dir / "main.py").write_text("def pip_main(): pass\n", encoding="utf-8")
        return FetchResult(
            source_dir=destination_dir,
            commit_sha=sha_v1,
            total_files=1,
            total_bytes=50,
        )

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch_ok):
        ws = manager.fetch_and_index_github_workspace(ws.id)

    units_before = ws.load_code_units()
    assert ws.metadata.commit_sha == sha_v1

    # Simulate fetch failure during refresh
    def mock_fetch_fail(*args, **kwargs):
        raise GithubFetchError("Connection reset by peer")

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch_fail):
        with pytest.raises(GithubFetchError, match="Connection reset"):
            manager.refresh_github_workspace(ws.id)

    # Re-retrieve workspace: previous usable state MUST be preserved!
    reloaded_ws = manager.get_workspace(ws.id)
    assert reloaded_ws.metadata.status == "ready"
    assert reloaded_ws.metadata.commit_sha == sha_v1
    assert reloaded_ws.load_code_units() == units_before
    assert (reloaded_ws.source_dir / "main.py").exists()


def test_interrupted_publication_recovery(temp_manager_env):
    """Verify that interrupted candidate folders are purged cleanly on restart/recovery."""
    manager = temp_manager_env["manager"]
    ws = manager.register_github_workspace("https://github.com/python/cpython")

    # Leave stale candidate artifacts as if a process crashed
    candidate_tmp = ws.workspace_dir / ".candidate_source_tmp"
    candidate_tmp.mkdir()
    (candidate_tmp / "junk.txt").write_text("stale", encoding="utf-8")

    staging_dir = ws.workspace_dir / ".staging_tmp"
    staging_dir.mkdir()
    (staging_dir / "partial.jsonl").write_text("stale", encoding="utf-8")

    # Getting workspace triggers _recover_workspace_staging
    reloaded_ws = manager.get_workspace(ws.id)
    assert not candidate_tmp.exists()
    assert not staging_dir.exists()


def test_remote_and_local_workspace_isolation(temp_manager_env):
    """Test remote GitHub workspace and local workspace remain fully isolated."""
    manager = temp_manager_env["manager"]
    root_tmp = temp_manager_env["root_tmp"]

    # 1. Local workspace
    local_dir = root_tmp / "local_repo"
    local_dir.mkdir()
    (local_dir / "local_service.py").write_text("def local_calc(): return 100\n", encoding="utf-8")
    local_ws = manager.register_workspace("local-ws", local_dir)
    manager.index_workspace("local-ws")

    # 2. GitHub workspace
    gh_ws = manager.register_github_workspace("https://github.com/myorg/cloud-service")

    def mock_fetch(identity, destination_dir, expected_parent, **kwargs):
        destination_dir.mkdir(parents=True, exist_ok=True)
        (destination_dir / "cloud.py").write_text("def cloud_deploy(): return 200\n", encoding="utf-8")
        return FetchResult(
            source_dir=destination_dir,
            commit_sha="d" * 40,
            total_files=1,
            total_bytes=50,
        )

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch):
        manager.fetch_and_index_github_workspace(gh_ws.id)

    # Verify separate paths
    assert local_ws.source_dir == local_dir.resolve()
    assert gh_ws.source_dir == (gh_ws.workspace_dir / "source").resolve()
    assert local_ws.workspace_dir != gh_ws.workspace_dir

    # Deleting GitHub workspace deletes its cloned source
    manager.delete_workspace(gh_ws.id)
    assert not gh_ws.workspace_dir.exists()

    # Deleting local workspace deletes index but leaves local_repo untouched
    manager.delete_workspace(local_ws.id)
    assert not local_ws.workspace_dir.exists()
    assert local_dir.exists()
    assert (local_dir / "local_service.py").exists()
