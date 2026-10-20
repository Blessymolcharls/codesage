"""CodeSage: Codebase intelligence and ingestion system."""

__version__ = "0.2.0"

from codesage.hybrid import HybridRetriever, HybridSearchResult, reciprocal_rank_fusion
from codesage.workspace import Workspace, WorkspaceManager, WorkspaceMetadata

__all__ = [
    "HybridRetriever",
    "HybridSearchResult",
    "reciprocal_rank_fusion",
    "Workspace",
    "WorkspaceManager",
    "WorkspaceMetadata",
]
