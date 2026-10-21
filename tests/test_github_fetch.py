"""Unit and security tests for GitHub repository fetcher and snapshot validation."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codesage.github import (
    FetchResult,
    GitHubRepoIdentity,
    GithubFetchError,
    OversizedRepositoryError,
    UnsafeRepositoryContentError,
    _safe_remove_dir,
    fetch_github_repository,
    parse_and_validate_github_url,
)


@pytest.fixture
def dummy_identity():
    return GitHubRepoIdentity(
        owner="test-owner",
        repo="test-repo",
        canonical_url="https://github.com/test-owner/test-repo",
        default_workspace_id="gh-test-owner-test-repo",
    )


def test_fetch_success_mocked(tmp_path: Path, dummy_identity: GitHubRepoIdentity):
    """Verify successful clone flow: commits extracted, .git pruned, size audited."""
    dest_dir = tmp_path / "target_source"
    expected_parent = tmp_path
    dummy_sha = "a" * 40

    def mock_subprocess_run(cmd, *args, **kwargs):
        # Emulate git clone
        if "clone" in cmd:
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / ".git").mkdir()
            (dest_dir / ".git" / "config").write_text("dummy", encoding="utf-8")
            (dest_dir / "index.py").write_text("def hello(): pass\n", encoding="utf-8")
            (dest_dir / "README.md").write_text("# Test Repo\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        elif "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=dummy_sha + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=mock_subprocess_run):
        result = fetch_github_repository(
            identity=dummy_identity,
            destination_dir=dest_dir,
            expected_parent=expected_parent,
        )

    assert isinstance(result, FetchResult)
    assert result.commit_sha == dummy_sha
    assert result.total_files == 2  # index.py and README.md (excluding .git)
    assert not (dest_dir / ".git").exists()  # .git purged
    assert (dest_dir / "index.py").exists()


def test_fetch_timeout_cleans_destination(tmp_path: Path, dummy_identity: GitHubRepoIdentity):
    """Verify timeout triggers clean abort and cleans up incomplete download."""
    dest_dir = tmp_path / "target_source"
    expected_parent = tmp_path

    def mock_timeout(cmd, *args, **kwargs):
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "partial.tmp").write_text("incomplete", encoding="utf-8")
        raise subprocess.TimeoutExpired(cmd, timeout=5)

    with patch("subprocess.run", side_effect=mock_timeout):
        with pytest.raises(GithubFetchError, match="timed out"):
            fetch_github_repository(
                identity=dummy_identity,
                destination_dir=dest_dir,
                expected_parent=expected_parent,
                timeout_seconds=5,
            )

    assert not dest_dir.exists()


def test_fetch_git_error_sanitizes_paths(tmp_path: Path, dummy_identity: GitHubRepoIdentity):
    """Verify git failure cleans up and sanitizes host paths."""
    dest_dir = tmp_path / "target_source"
    expected_parent = tmp_path

    def mock_failure(cmd, *args, **kwargs):
        dest_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(
            cmd,
            returncode=128,
            stdout="",
            stderr="fatal: could not access 'https://github.com/test-owner/test-repo': C:\\Users\\secret\\repo error",
        )

    with patch("subprocess.run", side_effect=mock_failure):
        with pytest.raises(GithubFetchError) as exc_info:
            fetch_github_repository(
                identity=dummy_identity,
                destination_dir=dest_dir,
                expected_parent=expected_parent,
            )

    assert "C:\\Users\\secret" not in str(exc_info.value)
    assert "[REDACTED_PATH]" in str(exc_info.value)
    assert not dest_dir.exists()


def test_fetch_invalid_commit_sha_aborts(tmp_path: Path, dummy_identity: GitHubRepoIdentity):
    """Verify non-hex or malformed commit SHA is rejected."""
    dest_dir = tmp_path / "target_source"
    expected_parent = tmp_path

    def mock_bad_sha(cmd, *args, **kwargs):
        if "clone" in cmd:
            dest_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        elif "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="not-a-valid-sha\n", stderr="")
        return subprocess.CompletedProcess(cmd, returncode=0)

    with patch("subprocess.run", side_effect=mock_bad_sha):
        with pytest.raises(GithubFetchError, match="Invalid commit SHA"):
            fetch_github_repository(
                identity=dummy_identity,
                destination_dir=dest_dir,
                expected_parent=expected_parent,
            )

    assert not dest_dir.exists()


def test_oversized_file_count_limit(tmp_path: Path, dummy_identity: GitHubRepoIdentity):
    """Verify repository exceeding max_files is rejected and cleaned up."""
    dest_dir = tmp_path / "target_source"
    expected_parent = tmp_path
    dummy_sha = "b" * 40

    def mock_many_files(cmd, *args, **kwargs):
        if "clone" in cmd:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                (dest_dir / f"file_{i}.txt").write_text("content", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode=0)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=dummy_sha + "\n")

    with patch("subprocess.run", side_effect=mock_many_files):
        with pytest.raises(OversizedRepositoryError, match="maximum file count limit"):
            fetch_github_repository(
                identity=dummy_identity,
                destination_dir=dest_dir,
                expected_parent=expected_parent,
                max_files=5,  # Threshold = 5 files, created 10
            )

    assert not dest_dir.exists()


def test_oversized_total_bytes_limit(tmp_path: Path, dummy_identity: GitHubRepoIdentity):
    """Verify repository exceeding total size is rejected and cleaned up."""
    dest_dir = tmp_path / "target_source"
    expected_parent = tmp_path
    dummy_sha = "c" * 40

    def mock_large_files(cmd, *args, **kwargs):
        if "clone" in cmd:
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "big.bin").write_bytes(b"X" * 2000)
            return subprocess.CompletedProcess(cmd, returncode=0)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=dummy_sha + "\n")

    with patch("subprocess.run", side_effect=mock_large_files):
        with pytest.raises(OversizedRepositoryError, match="maximum total size limit"):
            fetch_github_repository(
                identity=dummy_identity,
                destination_dir=dest_dir,
                expected_parent=expected_parent,
                max_total_bytes=1000,  # Threshold = 1000 bytes, created 2000
            )

    assert not dest_dir.exists()


def test_destination_outside_expected_parent_rejected(tmp_path: Path, dummy_identity: GitHubRepoIdentity):
    """Verify safe boundary enforcement blocks target paths outside designated parent."""
    outside_dir = tmp_path / "outside_dir"
    expected_parent = tmp_path / "workspaces" / "ws1"

    with pytest.raises(UnsafeRepositoryContentError, match="escapes expected parent"):
        fetch_github_repository(
            identity=dummy_identity,
            destination_dir=outside_dir,
            expected_parent=expected_parent,
        )


def test_safe_remove_dir_boundary_guard(tmp_path: Path):
    """Verify _safe_remove_dir refuses to delete directories outside expected parent."""
    legit_parent = tmp_path / "valid_parent"
    legit_parent.mkdir()
    child = legit_parent / "child"
    child.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()

    # Normal child is removed
    _safe_remove_dir(child, legit_parent)
    assert not child.exists()

    # Outside dir is refused
    with pytest.raises(UnsafeRepositoryContentError, match="outside expected parent"):
        _safe_remove_dir(outside, legit_parent)
    assert outside.exists()
