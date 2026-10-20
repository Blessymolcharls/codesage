"""Integration tests for CLI multi-repository and incremental indexing flags."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from codesage.workspace import WorkspaceManager


def test_cli_workspace_workflow(tmp_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    manager = WorkspaceManager(data_dir)
    ws_id = "test-cli-ws"

    # Cleanup any leftovers
    try:
        manager.delete_workspace(ws_id)
    except Exception:
        pass

    # 1. Create a dummy codebase
    repo_dir = tmp_path / "sample_cli_repo"
    repo_dir.mkdir()
    (repo_dir / "service.py").write_text(
        "def process_pdf_document(doc):\n    '''Process and convert PDF document.'''\n    return len(doc)\n",
        encoding="utf-8",
    )

    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}

    try:
        # 2. Test registration & full indexing via build_index.py
        res = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "build_index.py"),
                "--workspace",
                ws_id,
                "--repo-root",
                str(repo_dir),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode == 0, f"build_index failed: {res.stderr}"
        assert f"Workspace Indexing Summary: {ws_id}" in res.stdout

        ws = manager.get_workspace(ws_id)
        assert ws.metadata.status == "ready"
        assert ws.units_path.exists()

        # 3. Test build_graph.py with --workspace
        res_graph = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "build_graph.py"),
                "--workspace",
                ws_id,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res_graph.returncode == 0, f"build_graph failed: {res_graph.stderr}"
        assert ws.graph_path.exists()

        # 4. Test canonical graph immutability guard in build_graph.py
        res_canon = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "build_graph.py"),
                "--workspace",
                "default",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res_canon.returncode != 0
        assert "canonical research baseline relationship graph is immutable" in res_canon.stderr

        # 5. Test search.py with --workspace
        res_search = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "search.py"),
                "process_pdf_document",
                "--workspace",
                ws_id,
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res_search.returncode == 0, f"search failed: {res_search.stderr}"
        data = json.loads(res_search.stdout)
        assert data["workspace_id"] == ws_id
        assert data["results_count"] > 0
        assert any("process_pdf_document" in r["preview"] for r in data["results"])

        # Test rerank flag with tfidf
        res_rerank = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "search.py"),
                "process_pdf_document",
                "--workspace",
                ws_id,
                "--mode",
                "tfidf",
                "--rerank",
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res_rerank.returncode == 0, f"rerank search failed: {res_rerank.stderr}"
        data_rerank = json.loads(res_rerank.stdout)
        assert data_rerank["rerank_enabled"] is True

        # 6. Test incremental index via build_index.py --workspace <id> --incremental
        # Add a new file
        (repo_dir / "helper.py").write_text(
            "def helper_token_utility():\n    return 42\n",
            encoding="utf-8",
        )
        res_inc = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "build_index.py"),
                "--workspace",
                ws_id,
                "--incremental",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res_inc.returncode == 0, f"incremental failed: {res_inc.stderr}"
        assert "Incremental Indexing Summary" in res_inc.stdout
        assert "Files Added:    1" in res_inc.stdout

    finally:
        # Cleanup
        try:
            manager.delete_workspace(ws_id)
        except Exception:
            pass
