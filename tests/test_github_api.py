"""Integration tests for GitHub workspace endpoints in FastAPI backend."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from codesage.api import app, state
from codesage.github import (
    FetchResult,
    GithubFetchError,
    OversizedRepositoryError,
    UnsafeRepositoryContentError,
)
from codesage.workspace import WorkspaceManager


@pytest.fixture(scope="module")
def client():
    """Create test client within application lifecycle context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def temp_workspace_env(tmp_path, monkeypatch):
    """Isolate WorkspaceManager data directory during API testing."""
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    (test_data_dir / "workspaces").mkdir(parents=True, exist_ok=True)

    orig_mgr = state.workspace_manager
    state.workspace_manager = WorkspaceManager(test_data_dir)
    state.containers.clear()

    yield test_data_dir

    # Cleanup and restore
    state.workspace_manager = orig_mgr
    state.containers.clear()


def test_register_github_workspace_invalid_url(client: TestClient, temp_workspace_env):
    """Test registering with an invalid/malformed GitHub URL returns 400."""
    payload = {
        "url": "https://gitlab.com/owner/repo",
        "auto_index": False,
    }
    resp = client.post("/api/workspaces/github", json=payload)
    assert resp.status_code == 400
    assert "Unsupported host" in resp.json()["detail"]


def test_register_github_workspace_deceptive_url(client: TestClient, temp_workspace_env):
    """Test registering with userinfo or deceptive url returns 400."""
    payload = {
        "url": "https://user:pass@github.com/owner/repo",
        "auto_index": False,
    }
    resp = client.post("/api/workspaces/github", json=payload)
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "credential" in detail or "userinfo" in detail


def test_register_github_workspace_duplicate(client: TestClient, temp_workspace_env):
    """Test registering the exact same GitHub URL twice returns 409 Conflict."""
    payload = {
        "url": "https://github.com/psf/requests",
        "auto_index": False,
    }
    resp1 = client.post("/api/workspaces/github", json=payload)
    assert resp1.status_code == 201
    assert resp1.json()["id"] == "gh-psf-requests"

    resp2 = client.post("/api/workspaces/github", json=payload)
    assert resp2.status_code == 409
    assert "already registered" in resp2.json()["detail"]


def test_register_github_workspace_fetch_error(client: TestClient, temp_workspace_env):
    """Test git fetch failure returns 502 Bad Gateway."""
    with patch("codesage.workspace.fetch_github_repository") as mock_fetch:
        mock_fetch.side_effect = GithubFetchError("Git clone failed with code 128")
        payload = {
            "url": "https://github.com/psf/black",
            "auto_index": True,
        }
        resp = client.post("/api/workspaces/github", json=payload)
        assert resp.status_code == 502
        assert "Git clone failed" in resp.json()["detail"]


def test_register_github_workspace_timeout_error(client: TestClient, temp_workspace_env):
    """Test git fetch timeout returns 504 Gateway Timeout."""
    with patch("codesage.workspace.fetch_github_repository") as mock_fetch:
        mock_fetch.side_effect = GithubFetchError("Git operation timed out after 60s")
        payload = {
            "url": "https://github.com/psf/black",
            "auto_index": True,
        }
        resp = client.post("/api/workspaces/github", json=payload)
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]


def test_register_github_workspace_oversized_error(client: TestClient, temp_workspace_env):
    """Test oversized repository error returns 413 Payload Too Large."""
    with patch("codesage.workspace.fetch_github_repository") as mock_fetch:
        mock_fetch.side_effect = OversizedRepositoryError("Repository exceeds file count limit: 3000 files")
        payload = {
            "url": "https://github.com/huge/monorepo",
            "auto_index": True,
        }
        resp = client.post("/api/workspaces/github", json=payload)
        assert resp.status_code == 413
        assert "exceeds" in resp.json()["detail"]


def test_register_github_workspace_unsafe_content_error(client: TestClient, temp_workspace_env):
    """Test unsafe symlink error returns 422 Unprocessable Entity."""
    with patch("codesage.workspace.fetch_github_repository") as mock_fetch:
        mock_fetch.side_effect = UnsafeRepositoryContentError("Symlink points outside candidate root")
        payload = {
            "url": "https://github.com/attacker/symlink-escape",
            "auto_index": True,
        }
        resp = client.post("/api/workspaces/github", json=payload)
        assert resp.status_code == 422
        assert "Symlink points outside" in resp.json()["detail"]


def test_register_and_search_github_workspace(client: TestClient, temp_workspace_env, tmp_path):
    """Test end-to-end registration, indexing, details, and search on a remote GitHub workspace."""
    mock_src = tmp_path / "mock_github_repo"
    mock_src.mkdir(parents=True, exist_ok=True)
    (mock_src / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n    '''Add two integers.'''\n    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n    '''Multiply two integers.'''\n    return a * b\n",
        encoding="utf-8",
    )

    def mock_fetch(identity, destination_dir, **kwargs):
        shutil.copytree(mock_src, destination_dir, dirs_exist_ok=True)
        return FetchResult(
            source_dir=destination_dir,
            commit_sha="1111222233334444555566667777888899990000",
            total_files=1,
            total_bytes=200,
        )

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch):
        payload = {
            "url": "https://github.com/example/mathlib",
            "name": "MathLib Example",
            "auto_index": True,
        }
        resp = client.post("/api/workspaces/github", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "gh-example-mathlib"
        assert data["source_type"] == "github"
        assert data["commit_sha"] == "1111222233334444555566667777888899990000"
        assert data["unit_count"] >= 2
        assert data["status"] == "ready"

        # Verify details endpoint
        det_resp = client.get("/api/workspaces/gh-example-mathlib")
        assert det_resp.status_code == 200
        det_data = det_resp.json()
        assert det_data["id"] == "gh-example-mathlib"
        assert det_data["github_url"] == "https://github.com/example/mathlib"

        # Verify search works against this remote workspace
        search_payload = {
            "query": "add two integers",
            "model": "tfidf",
            "workspace_id": "gh-example-mathlib",
            "top_k": 3,
        }
        s_resp = client.post("/api/search", json=search_payload)
        assert s_resp.status_code == 200
        s_data = s_resp.json()
        assert s_data["workspace_id"] == "gh-example-mathlib"
        assert len(s_data["results"]) >= 1
        assert "add" in s_data["results"][0]["source_code"]


def test_refresh_github_workspace_lifecycle(client: TestClient, temp_workspace_env, tmp_path):
    """Test workspace refresh: up_to_date, updated, and failure preservation."""
    mock_src = tmp_path / "mock_refresh_repo"
    mock_src.mkdir(parents=True, exist_ok=True)
    (mock_src / "mod.py").write_text("def v1(): return 1\n", encoding="utf-8")

    current_sha = "aaaa000011112222333344445555666677778888"

    def mock_fetch(identity, destination_dir, **kwargs):
        shutil.copytree(mock_src, destination_dir, dirs_exist_ok=True)
        return FetchResult(
            source_dir=destination_dir,
            commit_sha=current_sha,
            total_files=1,
            total_bytes=100,
        )

    with patch("codesage.workspace.fetch_github_repository", side_effect=mock_fetch):
        # 1. Register & initial index
        reg_resp = client.post("/api/workspaces/github", json={"url": "https://github.com/example/refreshable"})
        assert reg_resp.status_code == 201
        ws_id = reg_resp.json()["id"]

        # 2. Refresh when HEAD is unchanged -> up_to_date
        ref1 = client.post(f"/api/workspaces/{ws_id}/refresh")
        assert ref1.status_code == 200
        assert ref1.json()["status"] == "up_to_date"
        assert ref1.json()["commit_sha"] == current_sha

        # 3. Simulate remote update
        current_sha = "bbbb000011112222333344445555666677778888"
        (mock_src / "mod.py").write_text("def v2(): return 2\n", encoding="utf-8")

        ref2 = client.post(f"/api/workspaces/{ws_id}/refresh")
        assert ref2.status_code == 200
        assert ref2.json()["status"] == "updated"
        assert ref2.json()["new_commit"] == current_sha

        # 4. Simulate fetch failure during refresh -> preserves current state
        with patch("codesage.workspace.fetch_github_repository", side_effect=GithubFetchError("Network down")):
            ref3 = client.post(f"/api/workspaces/{ws_id}/refresh")
            assert ref3.status_code == 502
            assert "Network down" in ref3.json()["detail"]

        # Check that workspace is still in ready state with commit SHA v2
        det = client.get(f"/api/workspaces/{ws_id}")
        assert det.status_code == 200
        assert det.json()["commit_sha"] == current_sha
