"""Unit tests for GitHub URL parsing, security validation, and identity normalization."""

import pytest

from codesage.github import (
    GitHubRepoIdentity,
    InvalidGithubUrlError,
    parse_and_validate_github_url,
)


def test_valid_canonical_github_urls():
    """Verify accepted canonical GitHub URLs and normalization."""
    # Standard URL
    id1 = parse_and_validate_github_url("https://github.com/facebook/react")
    assert id1.owner == "facebook"
    assert id1.repo == "react"
    assert id1.canonical_url == "https://github.com/facebook/react"
    assert id1.default_workspace_id == "gh-facebook-react"

    # URL with .git suffix
    id2 = parse_and_validate_github_url("https://github.com/facebook/react.git")
    assert id2.owner == "facebook"
    assert id2.repo == "react"
    assert id2.canonical_url == "https://github.com/facebook/react"
    assert id2.default_workspace_id == "gh-facebook-react"

    # URL with trailing slash
    id3 = parse_and_validate_github_url("https://github.com/facebook/react/")
    assert id3.owner == "facebook"
    assert id3.repo == "react"
    assert id3.canonical_url == "https://github.com/facebook/react"

    # URL with .git and trailing slash
    id4 = parse_and_validate_github_url("https://github.com/facebook/react.git/")
    assert id4.owner == "facebook"
    assert id4.repo == "react"
    assert id4.canonical_url == "https://github.com/facebook/react"

    # Allowed characters in owner and repo (dots, hyphens, underscores)
    id5 = parse_and_validate_github_url("https://github.com/pallets-eco/flask_sqlalchemy.org")
    assert id5.owner == "pallets-eco"
    assert id5.repo == "flask_sqlalchemy.org"
    assert id5.canonical_url == "https://github.com/pallets-eco/flask_sqlalchemy.org"
    assert id5.default_workspace_id == "gh-pallets-eco-flask-sqlalchemy-org"


def test_long_repository_name_id_truncation():
    """Verify long repo names are truncated safely with a deterministic hash suffix."""
    long_repo = "a" * 80
    url = f"https://github.com/someowner/{long_repo}"
    identity = parse_and_validate_github_url(url)
    assert len(identity.default_workspace_id) <= 64
    assert identity.default_workspace_id.startswith("gh-someowner-")
    # Repetition gives the exact same deterministic ID
    identity2 = parse_and_validate_github_url(url)
    assert identity.default_workspace_id == identity2.default_workspace_id


@pytest.mark.parametrize(
    "invalid_scheme_url",
    [
        "http://github.com/owner/repo",
        "git://github.com/owner/repo",
        "ssh://git@github.com/owner/repo",
        "file:///etc/passwd",
        "ftp://github.com/owner/repo",
        "github.com/owner/repo",  # Missing scheme
    ],
)
def test_reject_non_https_schemes(invalid_scheme_url: str):
    with pytest.raises(InvalidGithubUrlError, match="Invalid URL scheme|Only secure 'https://'"):
        parse_and_validate_github_url(invalid_scheme_url)


@pytest.mark.parametrize(
    "deceptive_host_url",
    [
        "https://gitlab.com/owner/repo",
        "https://bitbucket.org/owner/repo",
        "https://github.com.attacker.com/owner/repo",
        "https://evil-github.com/owner/repo",
        "https://attacker.com/github.com/owner/repo",
        "https://gist.github.com/owner/repo",
        "https://raw.githubusercontent.com/owner/repo",
        "https://api.github.com/owner/repo",
    ],
)
def test_reject_unsupported_or_deceptive_hosts(deceptive_host_url: str):
    with pytest.raises(InvalidGithubUrlError, match="Unsupported host|Only public repositories hosted on 'github.com'"):
        parse_and_validate_github_url(deceptive_host_url)


@pytest.mark.parametrize(
    "credentials_or_port_url",
    [
        "https://user:password@github.com/owner/repo",
        "https://token@github.com/owner/repo",
        "https://git:x-oauth-basic@github.com/owner/repo",
        "https://github.com:8080/owner/repo",
        "https://github.com:443/owner/repo",
    ],
)
def test_reject_credentials_and_ports(credentials_or_port_url: str):
    with pytest.raises(InvalidGithubUrlError, match="credentials or userinfo|Non-standard port"):
        parse_and_validate_github_url(credentials_or_port_url)


@pytest.mark.parametrize(
    "query_or_fragment_url",
    [
        "https://github.com/owner/repo?branch=main",
        "https://github.com/owner/repo#readme",
        "https://github.com/owner/repo?utm_source=test#section",
    ],
)
def test_reject_queries_and_fragments(query_or_fragment_url: str):
    with pytest.raises(InvalidGithubUrlError, match="Query strings are not permitted|URL fragments are not permitted"):
        parse_and_validate_github_url(query_or_fragment_url)


@pytest.mark.parametrize(
    "bad_path_url",
    [
        "https://github.com",
        "https://github.com/",
        "https://github.com/owner",
        "https://github.com/owner/",
        "https://github.com/owner/repo/tree/main",
        "https://github.com/owner/repo/blob/main/src/index.js",
        "https://github.com/owner/repo/issues",
        "https://github.com/owner/repo/releases/tag/v1.0.0",
        "https://github.com/..",
        "https://github.com/owner/..",
        "https://github.com/owner/.",
        "https://github.com/./repo",
        "https://github.com/-invalid-/repo",
        "https://github.com/owner/-invalid-name..",
    ],
)
def test_reject_invalid_paths_and_traversal(bad_path_url: str):
    with pytest.raises(InvalidGithubUrlError):
        parse_and_validate_github_url(bad_path_url)


def test_reject_empty_or_non_string():
    with pytest.raises(InvalidGithubUrlError):
        parse_and_validate_github_url("")
    with pytest.raises(InvalidGithubUrlError):
        parse_and_validate_github_url("   ")
    with pytest.raises(InvalidGithubUrlError):
        parse_and_validate_github_url(None)  # type: ignore
    with pytest.raises(InvalidGithubUrlError):
        parse_and_validate_github_url(123)  # type: ignore
