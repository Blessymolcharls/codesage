"""GitHub repository ingestion, URL validation, and secure fetch handling for CodeSage."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit


# GitHub username: 1-39 alphanumeric characters or single non-consecutive hyphens, not start/end with hyphen
GITHUB_OWNER_REGEX = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$")

# GitHub repository name: 1-100 characters containing alphanumeric, hyphens, underscores, dots
GITHUB_REPO_REGEX = re.compile(r"^[a-zA-Z0-9_.-]{1,100}$")

# Safe workspace ID regex matching codesage.workspace.SAFE_REPO_ID_REGEX
SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,63}$")


class GithubIngestionError(Exception):
    """Base exception for GitHub repository ingestion failures."""
    pass


class InvalidGithubUrlError(GithubIngestionError):
    """Raised when a supplied GitHub URL violates security, protocol, or syntax rules."""
    pass


class GithubFetchError(GithubIngestionError):
    """Raised when cloning, fetching, or checkout from GitHub fails."""
    pass


class OversizedRepositoryError(GithubIngestionError):
    """Raised when a repository exceeds file count or total size limits."""
    pass


class UnsafeRepositoryContentError(GithubIngestionError):
    """Raised when a fetched repository contains unsafe filesystem entries or symlink escapes."""
    pass


@dataclass(frozen=True)
class GitHubRepoIdentity:
    """Canonical identity of a validated public GitHub repository."""
    owner: str
    repo: str
    canonical_url: str
    default_workspace_id: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


def parse_and_validate_github_url(raw_url: str) -> GitHubRepoIdentity:
    """Parse, strictly validate, and normalize a public GitHub repository URL.

    Security & Validation Rules:
    1. Scheme must be strictly 'https'. Reject 'http', 'git', 'ssh', 'file', etc.
    2. Host must be strictly 'github.com' (no deceptive subdomains, external domains, or IP addresses).
    3. Userinfo / credentials (username:password@) are strictly rejected.
    4. Non-standard ports (e.g. :8080) are strictly rejected.
    5. Query strings and URL fragments are strictly rejected.
    6. Path must contain exactly two segments: '/{owner}/{repo}'.
    7. Trailing slashes and optional '.git' suffix are cleanly normalized.
    8. Owner and repository slugs must adhere to GitHub naming rules and reject path traversal ('..', '.').
    9. Generates a deterministic, safe, collision-resistant workspace ID.

    Args:
        raw_url: Candidate URL string.

    Returns:
        GitHubRepoIdentity containing validated owner, repo, canonical URL, and default workspace ID.

    Raises:
        InvalidGithubUrlError: If any security or structural validation fails.
    """
    if not isinstance(raw_url, str):
        raise InvalidGithubUrlError("Repository URL must be a string.")

    cleaned_url = raw_url.strip()
    if not cleaned_url:
        raise InvalidGithubUrlError("Repository URL cannot be empty.")

    # Reject null bytes or control characters
    if "\x00" in cleaned_url or any(ord(c) < 32 for c in cleaned_url):
        raise InvalidGithubUrlError("Repository URL contains illegal control characters.")

    try:
        parsed = urlsplit(cleaned_url)
    except Exception as e:
        raise InvalidGithubUrlError(f"Malformed URL structure: {e}") from e

    # 1. Scheme validation
    if parsed.scheme.lower() != "https":
        raise InvalidGithubUrlError(
            f"Invalid URL scheme '{parsed.scheme}'. Only secure 'https://' GitHub URLs are permitted."
        )

    # 2. Host validation & netloc inspection
    netloc = parsed.netloc.strip()
    if not netloc:
        raise InvalidGithubUrlError("Repository URL is missing hostname.")

    # Reject credentials/userinfo
    if "@" in netloc:
        raise InvalidGithubUrlError("Repository URLs containing credentials or userinfo are strictly prohibited.")

    # Reject unexpected ports
    if ":" in netloc:
        raise InvalidGithubUrlError(f"Non-standard port specified in URL netloc '{netloc}'. Only default HTTPS port 443 is permitted.")

    if netloc.lower() != "github.com":
        raise InvalidGithubUrlError(
            f"Unsupported host '{netloc}'. Only public repositories hosted on 'github.com' are permitted."
        )

    # 3. Query string and fragment rejection
    if parsed.query:
        raise InvalidGithubUrlError("Query strings are not permitted in GitHub repository URLs.")
    if parsed.fragment:
        raise InvalidGithubUrlError("URL fragments are not permitted in GitHub repository URLs.")

    # 4. Path segmentation and inspection
    raw_path = parsed.path.strip("/")
    if not raw_path:
        raise InvalidGithubUrlError("Missing repository path. Expected 'https://github.com/owner/repository'.")

    segments = [s for s in raw_path.split("/") if s]
    if len(segments) != 2:
        raise InvalidGithubUrlError(
            f"Invalid GitHub repository path '/{raw_path}'. Expected exactly two segments: 'owner/repository'."
        )

    owner, repo_segment = segments[0], segments[1]

    # Normalize optional .git suffix
    if repo_segment.lower().endswith(".git"):
        repo = repo_segment[:-4]
    else:
        repo = repo_segment

    # 5. Slug validation
    if not owner or not GITHUB_OWNER_REGEX.match(owner):
        raise InvalidGithubUrlError(
            f"Invalid GitHub owner/organization name '{owner}'. Must be 1-39 alphanumeric characters or hyphens."
        )

    if (
        not repo
        or not GITHUB_REPO_REGEX.match(repo)
        or repo in (".", "..")
        or repo.endswith(".")
    ):
        raise InvalidGithubUrlError(
            f"Invalid GitHub repository name '{repo}'. Must be 1-100 valid characters, cannot be '.' or '..', and cannot end with a period."
        )

    canonical_url = f"https://github.com/{owner}/{repo}"

    # 6. Deterministic, collision-resistant workspace ID generation
    # Format: gh-<owner>-<repo>, sanitized to alphanumeric and hyphens
    sanitized_owner = re.sub(r"[^a-zA-Z0-9-]", "-", owner).strip("-").lower()
    sanitized_repo = re.sub(r"[^a-zA-Z0-9-]", "-", repo).strip("-").lower()
    base_id = f"gh-{sanitized_owner}-{sanitized_repo}"

    # If length exceeds 56 chars, truncate and append SHA-256 fingerprint suffix for collision resistance
    if len(base_id) > 56:
        sha_suffix = hashlib.sha256(canonical_url.lower().encode("utf-8")).hexdigest()[:7]
        default_workspace_id = f"{base_id[:48].rstrip('-')}-{sha_suffix}"
    else:
        default_workspace_id = base_id

    # Fallback safety check
    if not SAFE_ID_REGEX.match(default_workspace_id):
        sha_suffix = hashlib.sha256(canonical_url.lower().encode("utf-8")).hexdigest()[:12]
        default_workspace_id = f"gh-{sha_suffix}"

    return GitHubRepoIdentity(
        owner=owner,
        repo=repo,
        canonical_url=canonical_url,
        default_workspace_id=default_workspace_id,
    )


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a successful, validated repository fetch."""
    source_dir: Path
    commit_sha: str
    total_files: int
    total_bytes: int


def _safe_remove_dir(target_dir: Path, expected_parent: Path) -> None:
    """Safely remove a directory ensuring it is strictly inside expected_parent.
    
    Prevents path traversal and recursive deletion outside the designated workspace.
    """
    if not target_dir.exists():
        return

    resolved_target = target_dir.resolve()
    resolved_parent = expected_parent.resolve()

    if resolved_target == resolved_parent or resolved_parent not in resolved_target.parents:
        raise UnsafeRepositoryContentError(
            f"Refusing to delete directory outside expected parent: {target_dir}"
        )

    if target_dir.is_symlink():
        target_dir.unlink()
    else:
        # On Windows, read-only files in .git/ or repo snapshots require chmod before rmtree
        def on_exc(fn, path, exc_info):
            try:
                os.chmod(path, 0o777)
                fn(path)
            except Exception:
                pass

        shutil.rmtree(target_dir, onexc=on_exc)


def fetch_github_repository(
    identity: GitHubRepoIdentity,
    destination_dir: Path,
    expected_parent: Path,
    timeout_seconds: int = 60,
    max_files: int = 2500,
    max_total_bytes: int = 50 * 1024 * 1024,
    max_file_size_bytes: int = 10 * 1024 * 1024,
) -> FetchResult:
    """Safely shallow-clone a public GitHub repository into a dedicated temporary destination.

    Security & Safety Controls:
    1. Validates remote identity and enforces https-only protocol policy.
    2. Invokes git using argument arrays with shell=False.
    3. Disables hooks (core.hooksPath="") and sets protocol.allow=never with https whitelist.
    4. Sets GIT_TERMINAL_PROMPT=0 and GIT_CONFIG_NOSYSTEM=1 to block interactive prompts.
    5. Performs a shallow single-branch clone (--depth 1 --single-branch --no-tags).
    6. Extracts and validates 40-character commit SHA (git rev-parse HEAD).
    7. Immediately removes .git metadata directory.
    8. Validates total file count, total size, and individual file size limits.
    9. Inspects filesystem entries and rejects symlinks pointing outside destination.
    10. Completely cleans up destination if any failure, timeout, or limit violation occurs.
    """
    resolved_dest = destination_dir.resolve()
    resolved_parent = expected_parent.resolve()

    if resolved_dest == resolved_parent or resolved_parent not in resolved_dest.parents:
        raise UnsafeRepositoryContentError(
            f"Destination directory escapes expected parent boundary: {destination_dir}"
        )

    # Clean existing destination if present
    if destination_dir.exists():
        _safe_remove_dir(destination_dir, expected_parent)

    destination_dir.parent.mkdir(parents=True, exist_ok=True)

    git_bin = shutil.which("git") or "git"

    # Strictly constrained git invocation
    cmd = [
        git_bin,
        "-c", "core.hooksPath=",
        "-c", "protocol.allow=never",
        "-c", "protocol.https.allow=always",
        "clone",
        "--depth", "1",
        "--single-branch",
        "--no-tags",
        "--",
        identity.canonical_url,
        str(destination_dir),
    ]

    env = {
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }

    try:
        res = subprocess.run(
            cmd,
            env=env,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        _safe_remove_dir(destination_dir, expected_parent)
        raise GithubFetchError(
            f"Git clone operation timed out after {timeout_seconds}s for {identity.canonical_url}"
        ) from e
    except Exception as e:
        _safe_remove_dir(destination_dir, expected_parent)
        raise GithubFetchError(f"Git execution failure: {e}") from e

    if res.returncode != 0:
        _safe_remove_dir(destination_dir, expected_parent)
        # Sanitize error message to avoid leaking local paths
        err_msg = res.stderr.strip() or res.stdout.strip() or f"exit code {res.returncode}"
        # Filter local path disclosure
        sanitized_err = re.sub(r"[a-zA-Z]:[\\/][^\s]+", "[REDACTED_PATH]", err_msg)
        raise GithubFetchError(f"Failed to clone repository: {sanitized_err}")

    # Extract and validate pinned commit SHA
    sha_cmd = [
        git_bin,
        "-c", "core.hooksPath=",
        "-C", str(destination_dir),
        "rev-parse",
        "HEAD",
    ]
    try:
        sha_res = subprocess.run(
            sha_cmd,
            env=env,
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        _safe_remove_dir(destination_dir, expected_parent)
        raise GithubFetchError(f"Failed to resolve commit SHA: {e}") from e

    if sha_res.returncode != 0:
        _safe_remove_dir(destination_dir, expected_parent)
        raise GithubFetchError("Failed to extract commit SHA from repository clone.")

    commit_sha = sha_res.stdout.strip()
    if not re.match(r"^[a-fA-F0-9]{40}$", commit_sha):
        _safe_remove_dir(destination_dir, expected_parent)
        raise GithubFetchError(f"Invalid commit SHA format returned by git: '{commit_sha}'")

    # Purge .git directory to ensure no repository hooks or hidden git objects remain
    git_dir = destination_dir / ".git"
    if git_dir.exists():
        _safe_remove_dir(git_dir, destination_dir)

    # Resource and filesystem safety audit
    total_files = 0
    total_bytes = 0

    for root_str, dirs, files in os.walk(str(destination_dir), followlinks=False):
        curr_dir = Path(root_str)

        # Check directory symlinks
        if curr_dir.is_symlink():
            try:
                res_d = curr_dir.resolve()
                if res_d != resolved_dest and resolved_dest not in res_d.parents:
                    _safe_remove_dir(destination_dir, expected_parent)
                    raise UnsafeRepositoryContentError(
                        f"Repository contains directory symlink escaping root: {curr_dir.name}"
                    )
            except Exception:
                _safe_remove_dir(destination_dir, expected_parent)
                raise UnsafeRepositoryContentError(f"Failed to resolve directory symlink safely: {curr_dir.name}")

        for f in files:
            file_path = curr_dir / f

            # Check file symlinks
            if file_path.is_symlink():
                try:
                    res_f = file_path.resolve()
                    if res_f != resolved_dest and resolved_dest not in res_f.parents:
                        _safe_remove_dir(destination_dir, expected_parent)
                        raise UnsafeRepositoryContentError(
                            f"Repository contains file symlink escaping root: {f}"
                        )
                except Exception:
                    _safe_remove_dir(destination_dir, expected_parent)
                    raise UnsafeRepositoryContentError(f"Failed to resolve file symlink safely: {f}")

            total_files += 1
            if total_files > max_files:
                _safe_remove_dir(destination_dir, expected_parent)
                raise OversizedRepositoryError(
                    f"Repository exceeds maximum file count limit ({max_files} files)."
                )

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > max_file_size_bytes:
                # If a single file exceeds the fetch ceiling
                _safe_remove_dir(destination_dir, expected_parent)
                raise OversizedRepositoryError(
                    f"Repository contains single file '{f}' exceeding size limit ({max_file_size_bytes // 1024} KB)."
                )

            total_bytes += size
            if total_bytes > max_total_bytes:
                _safe_remove_dir(destination_dir, expected_parent)
                raise OversizedRepositoryError(
                    f"Repository exceeds maximum total size limit ({max_total_bytes // (1024*1024)} MB)."
                )

    return FetchResult(
        source_dir=destination_dir,
        commit_sha=commit_sha,
        total_files=total_files,
        total_bytes=total_bytes,
    )
