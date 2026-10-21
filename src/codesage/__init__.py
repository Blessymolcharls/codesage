"""CodeSage: Codebase intelligence and ingestion system."""

__version__ = "0.2.0"

from codesage.github import (
    GitHubRepoIdentity,
    GithubFetchError,
    GithubIngestionError,
    InvalidGithubUrlError,
    OversizedRepositoryError,
    UnsafeRepositoryContentError,
    fetch_github_repository,
    parse_and_validate_github_url,
)
from codesage.hybrid import HybridRetriever, HybridSearchResult, reciprocal_rank_fusion
from codesage.workspace import (
    DuplicateWorkspaceError,
    InvalidWorkspaceIdError,
    UnsafePathError,
    Workspace,
    WorkspaceError,
    WorkspaceManager,
    WorkspaceMetadata,
    WorkspaceNotFoundError,
)

__all__ = [
    "HybridRetriever",
    "HybridSearchResult",
    "reciprocal_rank_fusion",
    "Workspace",
    "WorkspaceManager",
    "WorkspaceMetadata",
    "WorkspaceError",
    "WorkspaceNotFoundError",
    "InvalidWorkspaceIdError",
    "UnsafePathError",
    "DuplicateWorkspaceError",
    "GitHubRepoIdentity",
    "GithubIngestionError",
    "InvalidGithubUrlError",
    "GithubFetchError",
    "OversizedRepositoryError",
    "UnsafeRepositoryContentError",
    "parse_and_validate_github_url",
    "fetch_github_repository",
]
