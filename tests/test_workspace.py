"""
Tests for multi-repository workspace management, filesystem security,
atomicity, cache isolation, and canonical benchmark protection.
"""

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from codesage.api import app, sanitize_error_message, state
from codesage.workspace import (
    WorkspaceManager,
    InvalidWorkspaceIdError,
    WorkspaceNotFoundError,
    WorkspaceError,
    UnsafePathError,
)
from codesage.graph import RepositoryGraphBuilder
from codesage.retrieval import TfidfRetriever


@pytest.fixture
def temp_workspace_env():
    """Create a temporary environment with two independent repositories and isolated data dir."""
    with tempfile.TemporaryDirectory() as base_tmp:
        base_path = Path(base_tmp).resolve()
        data_dir = base_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        repo_a = base_path / "repo_alpha"
        repo_a.mkdir(parents=True, exist_ok=True)
        (repo_a / "utils.py").write_text(
            "def calculate_tax(amount):\n"
            "    '''Calculate standard rate tax for alpha.'''\n"
            "    return amount * 0.15\n\n"
            "class InvoiceProcessor:\n"
            "    '''Processes invoices specifically for Alpha system.'''\n"
            "    def process(self):\n"
            "        return calculate_tax(100)\n",
            encoding="utf-8"
        )

        repo_b = base_path / "repo_beta"
        repo_b.mkdir(parents=True, exist_ok=True)
        (repo_b / "utils.py").write_text(
            "def calculate_tax(amount):\n"
            "    '''Calculate special reduced tax for beta.'''\n"
            "    return amount * 0.05\n\n"
            "class InvoiceProcessor:\n"
            "    '''Processes invoices specifically for Beta system.'''\n"
            "    def process(self):\n"
            "        return calculate_tax(50)\n",
            encoding="utf-8"
        )

        manager = WorkspaceManager(data_dir=data_dir)

        yield {
            "base_path": base_path,
            "data_dir": data_dir,
            "repo_a": repo_a,
            "repo_b": repo_b,
            "manager": manager,
        }


def test_workspace_id_security_edge_cases(temp_workspace_env):
    """Test comprehensive path traversal, reserved names, and illegal character rejection in workspace IDs."""
    manager = temp_workspace_env["manager"]

    # Valid IDs
    assert manager.validate_workspace_id("my-repo") == "my-repo"
    assert manager.validate_workspace_id("repo_123") == "repo_123"
    assert manager.validate_workspace_id("project.v2") == "project.v2"
    assert manager.validate_workspace_id("RepoA") == "RepoA"

    # Directory traversal attempts
    invalid_ids = [
        "../escaped",
        "..",
        ".",
        "./nested",
        "sub/dir",
        "repo\\backslash",
        "repo/forward",
        "/root_dir",
        "C:\\Windows",
        "repo:colon",
        "repo*wildcard",
        "repo?question",
        "repo|pipe",
        "",
        "   ",
        "repo\x00null",
        "trailing_dot.",
        ".leading_dot",
        "-leading_dash",
        "_leading_underscore",
        # Reserved device names
        "con",
        "aux",
        "nul",
        "prn",
        "com1",
        "com9",
        "lpt1",
        "default",
        "docucharm",
    ]

    for bad_id in invalid_ids:
        with pytest.raises(InvalidWorkspaceIdError):
            manager.validate_workspace_id(bad_id)


def test_independent_repositories_isolation(temp_workspace_env):
    """Test that two independent repositories with identical file and symbol names are completely isolated."""
    manager = temp_workspace_env["manager"]
    repo_a = temp_workspace_env["repo_a"]
    repo_b = temp_workspace_env["repo_b"]

    # Register and index repo-a
    manager.register_workspace(
        repo_id="repo-alpha",
        repo_path=repo_a,
        name="Alpha Project",
    )
    ws_a = manager.index_workspace("repo-alpha")
    assert ws_a.metadata.status == "ready"
    assert ws_a.metadata.unit_count > 0

    # Register and index repo-b
    manager.register_workspace(
        repo_id="repo-beta",
        repo_path=repo_b,
        name="Beta Project",
    )
    ws_b = manager.index_workspace("repo-beta")
    assert ws_b.metadata.status == "ready"
    assert ws_b.metadata.unit_count > 0

    # Verify storage isolation
    assert ws_a.workspace_dir != ws_b.workspace_dir
    assert ws_a.units_path.exists()
    assert ws_b.units_path.exists()
    assert ws_a.cache_dir.exists()
    assert ws_b.cache_dir.exists()
    assert ws_a.cache_dir != ws_b.cache_dir

    # Verify code units isolation
    units_a = ws_a.load_code_units()
    units_b = ws_b.load_code_units()

    ids_a = {u["unit_id"] for u in units_a}
    ids_b = {u["unit_id"] for u in units_b}

    # Verify symbols content isolation
    tax_a = next(u for u in units_a if u.get("symbol_name") == "calculate_tax")
    tax_b = next(u for u in units_b if u.get("symbol_name") == "calculate_tax")
    assert "0.15" in (tax_a.get("source_text") or tax_a.get("source_code", ""))
    assert "0.05" in (tax_b.get("source_text") or tax_b.get("source_code", ""))

    # Search in alpha should never return beta code
    searcher_a = TfidfRetriever()
    searcher_a.fit(units_a)
    results_a = searcher_a.search("Alpha system invoice", top_k=5)
    assert len(results_a) > 0
    for r in results_a:
        assert "Beta" not in r.source_code
        assert "0.05" not in r.source_code

    # Search in beta should never return alpha code
    searcher_b = TfidfRetriever()
    searcher_b.fit(units_b)
    results_b = searcher_b.search("Beta system invoice", top_k=5)
    assert len(results_b) > 0
    for r in results_b:
        assert "Alpha" not in r.source_code
        assert "0.15" not in r.source_code

    # Graph isolation
    graph_a = ws_a.load_graph()
    graph_b = ws_b.load_graph()
    assert graph_a is not None and graph_b is not None
    graph_a_unit_ids = {nid.replace("unit:", "") for nid in graph_a.nodes.keys() if nid.startswith("unit:")}
    graph_b_unit_ids = {nid.replace("unit:", "") for nid in graph_b.nodes.keys() if nid.startswith("unit:")}
    assert graph_a_unit_ids.issubset(ids_a)
    assert graph_b_unit_ids.issubset(ids_b)


def test_empty_and_malformed_repositories(temp_workspace_env):
    """Test indexing empty repos and repos with malformed files gracefully without crashes."""
    manager = temp_workspace_env["manager"]
    base_path = temp_workspace_env["base_path"]

    # 1. Empty repository
    empty_repo = base_path / "empty_repo"
    empty_repo.mkdir()
    manager.register_workspace("empty-repo", empty_repo)
    ws_empty = manager.index_workspace("empty-repo")
    assert ws_empty.metadata.status == "ready"
    assert ws_empty.metadata.unit_count == 0
    assert ws_empty.load_code_units() == []

    # Search against empty index returns empty list without error
    searcher = TfidfRetriever()
    searcher.fit([])
    results = searcher.search("anything", top_k=5)
    assert results == []

    # 2. Malformed files
    malformed_repo = base_path / "malformed_repo"
    malformed_repo.mkdir()
    (malformed_repo / "broken.py").write_text("def class for while = syntax error !?@#", encoding="utf-8")
    (malformed_repo / "valid.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    manager.register_workspace("malformed-repo", malformed_repo)
    ws_malformed = manager.index_workspace("malformed-repo")
    assert ws_malformed.metadata.status == "ready"
    units = ws_malformed.load_code_units()
    assert any(u.get("symbol_name") == "hello" for u in units)


def test_incremental_indexing(temp_workspace_env):
    """Test incremental indexing detects added, modified, and deleted files accurately."""
    manager = temp_workspace_env["manager"]
    repo_a = temp_workspace_env["repo_a"]

    manager.register_workspace("repo-alpha", repo_a)
    ws_init = manager.index_workspace("repo-alpha")
    # utils.py contains: calculate_tax (function), InvoiceProcessor (class), process (method) = 3 units
    assert ws_init.metadata.unit_count == 3

    # 1. Add a file
    (repo_a / "extra.py").write_text("def extra_function():\n    return 42\n", encoding="utf-8")
    ws_inc1, stats1 = manager.incremental_index("repo-alpha")
    assert stats1["mode"] == "incremental"
    assert stats1["added"] == 1
    assert ws_inc1.metadata.unit_count == 4

    units1 = ws_inc1.load_code_units()
    names1 = {u.get("symbol_name") for u in units1}
    assert "extra_function" in names1

    # 2. Modify a file
    (repo_a / "extra.py").write_text("def extra_function_renamed():\n    return 99\n", encoding="utf-8")
    ws_inc2, stats2 = manager.incremental_index("repo-alpha")
    assert stats2["mode"] == "incremental"
    assert stats2["modified"] == 1
    assert ws_inc2.metadata.unit_count == 4
    units2 = ws_inc2.load_code_units()
    names2 = {u.get("symbol_name") for u in units2}
    assert "extra_function_renamed" in names2
    assert "extra_function" not in names2

    # 3. Delete a file
    (repo_a / "extra.py").unlink()
    ws_inc3, stats3 = manager.incremental_index("repo-alpha")
    assert stats3["mode"] == "incremental"
    assert stats3["deleted"] == 1
    assert ws_inc3.metadata.unit_count == 3
    units3 = ws_inc3.load_code_units()
    names3 = {u.get("symbol_name") for u in units3}
    assert "extra_function_renamed" not in names3


def test_incremental_indexing_rename_and_symbol_changes(temp_workspace_env):
    """Test renaming a file removes stale units and registers units under the new path."""
    manager = temp_workspace_env["manager"]
    repo_a = temp_workspace_env["repo_a"]

    manager.register_workspace("repo-rename-test", repo_a)
    manager.index_workspace("repo-rename-test")

    # Rename utils.py -> math_tools.py and change symbol calculate_tax -> compute_alpha_tax
    (repo_a / "utils.py").unlink()
    (repo_a / "math_tools.py").write_text(
        "def compute_alpha_tax(amount):\n"
        "    '''New renamed taxation function.'''\n"
        "    return amount * 0.20\n",
        encoding="utf-8"
    )

    ws, stats = manager.incremental_index("repo-rename-test")
    assert stats["mode"] == "incremental"
    assert stats["deleted"] == 1
    assert stats["added"] == 1

    units = ws.load_code_units()
    paths = {u["repository_relative_path"] for u in units}
    symbols = {u.get("symbol_name") for u in units}

    assert "utils.py" not in paths
    assert "math_tools.py" in paths
    assert "calculate_tax" not in symbols
    assert "compute_alpha_tax" in symbols


def test_incremental_indexing_changed_imports_and_graph(temp_workspace_env):
    """Test that modifying imports between files updates graph edges and removes stale targets."""
    manager = temp_workspace_env["manager"]
    base_path = temp_workspace_env["base_path"]

    dep_repo = base_path / "dep_repo"
    dep_repo.mkdir(parents=True, exist_ok=True)
    (dep_repo / "producer.py").write_text(
        "def produce_item():\n"
        "    return 'item'\n",
        encoding="utf-8"
    )
    (dep_repo / "consumer.py").write_text(
        "from producer import produce_item\n\n"
        "def consume():\n"
        "    return produce_item()\n",
        encoding="utf-8"
    )

    manager.register_workspace("dep-test", dep_repo)
    ws = manager.index_workspace("dep-test")
    graph1 = ws.load_graph()
    assert graph1 is not None

    # Modify consumer.py to remove the import and call
    (dep_repo / "consumer.py").write_text(
        "def consume():\n"
        "    return 'standalone'\n",
        encoding="utf-8"
    )

    ws_inc, _ = manager.incremental_index("dep-test")
    graph2 = ws_inc.load_graph()
    assert graph2 is not None

    # Verify consumer node no longer has outgoing calls to produce_item
    consumer_unit = next(u for u in ws_inc.load_code_units() if u.get("symbol_name") == "consume")
    neighbors = graph2.get_neighbors(f"unit:{consumer_unit['unit_id']}", direction="outgoing")
    target_names = [target for target, rel, _ in neighbors if "produce_item" in target]
    assert len(target_names) == 0


def test_transactional_indexing_atomicity_on_failure(temp_workspace_env):
    """Test that failure during indexing or graph construction never corrupts existing valid index."""
    manager = temp_workspace_env["manager"]
    repo_a = temp_workspace_env["repo_a"]

    manager.register_workspace("repo-alpha", repo_a)
    ws_initial = manager.index_workspace("repo-alpha")
    initial_units_sha = hashlib.sha256(ws_initial.units_path.read_bytes()).hexdigest()
    initial_graph_sha = hashlib.sha256(ws_initial.graph_path.read_bytes()).hexdigest()

    # Simulate an unhandled crash during graph construction
    with patch.object(RepositoryGraphBuilder, "build", side_effect=RuntimeError("Simulated graph crash")):
        with pytest.raises(WorkspaceError):
            manager.index_workspace("repo-alpha")

    # Staging was aborted; original index artifacts must remain 100% identical and uncorrupted
    ws_after = manager.get_workspace("repo-alpha")
    after_units_sha = hashlib.sha256(ws_after.units_path.read_bytes()).hexdigest()
    after_graph_sha = hashlib.sha256(ws_after.graph_path.read_bytes()).hexdigest()

    assert after_units_sha == initial_units_sha
    assert after_graph_sha == initial_graph_sha
    assert ws_after.load_code_units() != []


def test_concurrency_per_workspace_locking(temp_workspace_env):
    """Test that concurrent indexing operations on the same workspace serialize safely without file corruption."""
    manager = temp_workspace_env["manager"]
    repo_a = temp_workspace_env["repo_a"]

    manager.register_workspace("repo-concurrent", repo_a)

    errors = []

    def index_worker():
        try:
            manager.index_workspace("repo-concurrent")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=index_worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    ws = manager.get_workspace("repo-concurrent")
    assert ws.metadata.status == "ready"
    assert ws.metadata.unit_count == 3
    # Verify index and graph are valid JSON/JSONL
    assert len(ws.load_code_units()) == 3
    assert ws.load_graph() is not None


def test_workspace_deletion_preserves_source(temp_workspace_env):
    """Test deleting a workspace index deletes generated artifacts but leaves repo source untouched."""
    manager = temp_workspace_env["manager"]
    repo_a = temp_workspace_env["repo_a"]

    manager.register_workspace("repo-alpha", repo_a)
    manager.index_workspace("repo-alpha")

    ws = manager.get_workspace("repo-alpha")
    assert ws.workspace_dir.exists()
    assert (repo_a / "utils.py").exists()

    # Delete workspace
    manager.delete_workspace("repo-alpha")

    # Artifacts removed
    assert not ws.workspace_dir.exists()
    with pytest.raises(WorkspaceNotFoundError):
        manager.get_workspace("repo-alpha")

    # Source files intact
    assert (repo_a / "utils.py").exists()
    assert "InvoiceProcessor" in (repo_a / "utils.py").read_text(encoding="utf-8")


def test_canonical_benchmark_protection(temp_workspace_env):
    """Verify that canonical research baseline cannot be registered, reindexed, or deleted."""
    manager = temp_workspace_env["manager"]
    repo_a = temp_workspace_env["repo_a"]

    # 1. Cannot register reserved canonical IDs
    with pytest.raises(InvalidWorkspaceIdError):
        manager.register_workspace("default", repo_a)

    with pytest.raises(InvalidWorkspaceIdError):
        manager.register_workspace("docucharm", repo_a)

    # 2. Cannot delete canonical baseline
    with pytest.raises(WorkspaceError):
        manager.delete_workspace("default")

    with pytest.raises(WorkspaceError):
        manager.delete_workspace("docucharm")

    # 3. Cannot reindex canonical baseline
    with pytest.raises(WorkspaceError):
        manager.index_workspace("default")

    with pytest.raises(WorkspaceError):
        manager.incremental_index("default")


def test_target_repo_path_containment_and_symlinks(temp_workspace_env):
    """Test boundary containment preventing registration of internal data dir as a codebase."""
    manager = temp_workspace_env["manager"]
    data_dir = temp_workspace_env["data_dir"]

    # Refuse to register data_dir or workspaces_dir as codebases
    with pytest.raises(UnsafePathError):
        manager.register_workspace("invalid-data-target", data_dir)

    with pytest.raises(UnsafePathError):
        manager.register_workspace("invalid-ws-target", manager.workspaces_dir)

    # Refuse to register subdirectories inside data_dir or workspaces_dir
    sub_dir = data_dir / "cache"
    sub_dir.mkdir(exist_ok=True)
    with pytest.raises(UnsafePathError):
        manager.register_workspace("invalid-data-sub", sub_dir)

    ws_sub = manager.workspaces_dir / "other_repo"
    ws_sub.mkdir(exist_ok=True)
    with pytest.raises(UnsafePathError):
        manager.register_workspace("invalid-ws-sub", ws_sub)


def test_error_message_path_sanitization():
    """Verify that host filesystem paths are redacted from error messages."""
    win_path = r"Error reading file C:\Users\bless\AppData\Local\Temp\repo\secret.py: access denied"
    posix_path = "Failed at /home/developer/git/project/config.yaml: not found"

    sanitized_win = sanitize_error_message(win_path)
    sanitized_posix = sanitize_error_message(posix_path)

    assert "C:\\Users" not in sanitized_win
    assert "[REDACTED_PATH]" in sanitized_win
    assert "/home/developer" not in sanitized_posix
    assert "[REDACTED_PATH]" in sanitized_posix


def test_api_multi_workspace_endpoints(temp_workspace_env):
    """Test FastAPI workspace endpoints and workspace-scoped search/compare."""
    manager = temp_workspace_env["manager"]
    repo_a = temp_workspace_env["repo_a"]

    orig_manager = state.workspace_manager
    orig_containers = state.containers
    state.workspace_manager = manager
    state.containers = {}

    try:
        client = TestClient(app)

        # 1. GET /api/workspaces
        resp = client.get("/api/workspaces")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert any(w["id"] == "default" for w in data)

        # 2. POST /api/workspaces - Register new workspace
        reg_resp = client.post("/api/workspaces", json={
            "id": "api-test-alpha",
            "repo_path": str(repo_a),
            "name": "API Test Alpha"
        })
        assert reg_resp.status_code == 201
        reg_data = reg_resp.json()
        assert reg_data["id"] == "api-test-alpha"

        # 3. POST /api/workspaces/{id}/index - Run indexing
        idx_resp = client.post("/api/workspaces/api-test-alpha/index", json={"incremental": False})
        assert idx_resp.status_code == 200
        idx_data = idx_resp.json()
        assert idx_data["status"] == "ready"
        assert idx_data["unit_count"] == 3

        # 4. GET /api/workspaces/{id}
        ws_resp = client.get("/api/workspaces/api-test-alpha")
        assert ws_resp.status_code == 200
        assert ws_resp.json()["unit_count"] == 3

        # 5. POST /api/search with workspace_id
        search_resp = client.post("/api/search", json={
            "query": "calculate_tax",
            "model": "tfidf",
            "top_k": 5,
            "workspace_id": "api-test-alpha"
        })
        assert search_resp.status_code == 200
        s_data = search_resp.json()
        assert len(s_data["results"]) > 0
        assert s_data["results"][0]["symbol_name"] == "calculate_tax"

        # 6. POST /api/compare with workspace_id
        cmp_resp = client.post("/api/compare", json={
            "query": "invoice process",
            "top_k": 3,
            "workspace_id": "api-test-alpha"
        })
        assert cmp_resp.status_code == 200
        cmp_data = cmp_resp.json()
        assert "tfidf" in cmp_data["systems"]

        # 7. Non-existent workspace search yields 404
        bad_resp = client.post("/api/search", json={
            "query": "anything",
            "workspace_id": "non-existent-workspace"
        })
        assert bad_resp.status_code == 404

        # 8. Malformed workspace ID yields 400
        malformed_resp = client.get("/api/workspaces/..")
        assert malformed_resp.status_code in (400, 404)

        # 9. Attempted deletion of default yields 400
        del_default_resp = client.delete("/api/workspaces/default")
        assert del_default_resp.status_code == 400

        # 10. DELETE /api/workspaces/{id}
        del_resp = client.delete("/api/workspaces/api-test-alpha")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

    finally:
        state.workspace_manager = orig_manager
        state.containers = orig_containers
