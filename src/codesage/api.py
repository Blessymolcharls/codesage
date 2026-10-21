"""FastAPI backend server wrapping CodeSage code search, multi-repository workspaces, and evaluation."""

import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from codesage.code_semantic import CodeTransformerRetriever
from codesage.graph import RepositoryGraph, RepositoryGraphBuilder
from codesage.graph_reranker import GraphContextReranker
from codesage.hybrid import HybridRetriever, HybridSearchResult
from codesage.retrieval import SearchResult, TfidfRetriever, load_code_units_from_jsonl
from codesage.semantic import SemanticRetriever
from codesage.github import (
    GithubFetchError,
    GithubIngestionError,
    InvalidGithubUrlError,
    OversizedRepositoryError,
    UnsafeRepositoryContentError,
)
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

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "code_units.jsonl"
GRAPH_PATH = DATA_DIR / "relationship_graph.json"
CACHE_DIR = DATA_DIR / "cache"
BENCHMARK_JSON_PATH = DATA_DIR / "benchmark_comparison.json"
STUDY_JSON_PATH = DATA_DIR / "experiments" / "study_results.json"
GRAPH_RERANK_JSON_PATH = DATA_DIR / "experiments" / "graph_reranking_results.json"
HYBRID_JSON_PATH = DATA_DIR / "experiments" / "hybrid_fusion_results.json"


def sanitize_error_message(msg: str) -> str:
    """Mask absolute filesystem paths in error messages to prevent host information disclosure."""
    # Mask Windows drive paths (e.g. C:\... or C:/...)
    sanitized = re.sub(r"[a-zA-Z]:[\\/][^:\s\"']+", "[REDACTED_PATH]", str(msg))
    # Mask POSIX paths
    sanitized = re.sub(r"/(Users|home|etc|var|tmp|root)/[^\s\"']+", "[REDACTED_PATH]", sanitized)
    return sanitized


# Pydantic Request & Response Schemas
class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural-language or keyword search query",
        examples=["merge multiple PDF files"],
    )
    model: str = Field(
        default="tfidf",
        description="Retriever model: 'tfidf', 'semantic', 'code', 'hybrid', 'tfidf_graph', 'semantic_graph', 'code_graph', 'hybrid_graph'",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of code units to return (1-50)",
    )
    rerank: bool = Field(
        default=False,
        description="Enable graph-context neighborhood affinity reranking",
    )
    workspace_id: str = Field(
        default="default",
        description="Target workspace identifier to query (default: 'default')",
    )


class CompareRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural-language search query to benchmark across all engines",
        examples=["cleanup temporary files"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to retrieve per system (1-50)",
    )
    with_graph: bool = Field(
        default=False,
        description="Include graph-reranked counterparts in comparison",
    )
    workspace_id: str = Field(
        default="default",
        description="Target workspace identifier to benchmark (default: 'default')",
    )


class SearchResultItem(BaseModel):
    unit_id: str
    score: float
    rel_path: str
    language: str
    symbol_name: Optional[str] = None
    unit_type: str
    start_line: int
    end_line: int
    extraction_method: str
    preview: str
    source_code: str
    # Hybrid provenance fields
    model_ranks: Optional[Dict[str, int]] = None
    rrf_score: Optional[float] = None
    graph_boost: Optional[float] = None
    graph_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    model: str
    top_k: int
    rerank_enabled: bool
    latency_ms: float
    total_results: int
    workspace_id: str = "default"
    results: List[SearchResultItem]


class CompareSystemResult(BaseModel):
    name: str
    model_slug: str
    latency_ms: float
    results: List[SearchResultItem]


class CompareResponse(BaseModel):
    query: str
    top_k: int
    total_latency_ms: float
    workspace_id: str = "default"
    systems: Dict[str, CompareSystemResult]


class ModelInfo(BaseModel):
    id: str
    name: str
    category: str
    description: str
    dimensions: str
    parameters: str
    weight_mb: float
    supports_graph_reranking: bool
    is_available: bool = True


class RegisterWorkspaceRequest(BaseModel):
    id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique alphanumeric workspace slug (1-64 chars)",
        examples=["flutter-sample"],
    )
    repo_path: str = Field(
        ...,
        min_length=1,
        description="Local filesystem directory to target repository",
        examples=["../DocuCharm"],
    )
    name: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional human-readable display title",
    )


class RegisterGithubWorkspaceRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Public GitHub repository URL (e.g. 'https://github.com/owner/repo')",
        examples=["https://github.com/pallets/click"],
    )
    name: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional human-readable display title",
    )
    auto_index: bool = Field(
        default=True,
        description="Automatically fetch, scan, and index immediately after registration",
    )
    max_files: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional limit on number of source files to scan",
    )
    max_file_size_kb: int = Field(
        default=512,
        ge=1,
        le=10240,
        description="Maximum file size in kilobytes",
    )


class IndexWorkspaceRequest(BaseModel):
    incremental: bool = Field(
        default=False,
        description="Perform incremental re-index if prior index exists",
    )
    max_files: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional limit on number of source files to scan",
    )
    max_file_size_kb: int = Field(
        default=512,
        ge=1,
        le=10240,
        description="Maximum file size in kilobytes",
    )
    max_unit_lines: int = Field(
        default=250,
        ge=10,
        le=5000,
        description="Maximum line threshold before chunk splitting",
    )


# Workspace Retriever Container (per-workspace isolated state)
class WorkspaceContainer:
    """Holds loaded code units, graph, and fitted retrievers for a specific workspace."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.code_units: List[Dict[str, Any]] = []
        self.code_units_by_id: Dict[str, Dict[str, Any]] = {}
        self.graph: Optional[RepositoryGraph] = None

        self.tfidf_retriever: Optional[TfidfRetriever] = None
        self.semantic_retriever: Optional[SemanticRetriever] = None
        self.code_retriever: Optional[CodeTransformerRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None

        # Graph Rerankers
        self.tfidf_reranker: Optional[GraphContextReranker] = None
        self.semantic_reranker: Optional[GraphContextReranker] = None
        self.code_reranker: Optional[GraphContextReranker] = None

        self.models_loaded: List[str] = []
        self.is_loaded: bool = False

    def load(self, neural_available: bool = True) -> None:
        """Load code units, graph, and fit available retrieval engines."""
        if self.is_loaded:
            return

        self.code_units = self.workspace.load_code_units()
        self.code_units_by_id = {u["unit_id"]: u for u in self.code_units}
        self.graph = self.workspace.load_graph()

        # 1. Fit TF-IDF
        if self.code_units:
            tfidf = TfidfRetriever(include_metadata=True)
            tfidf.fit(self.code_units)
            self.tfidf_retriever = tfidf
            self.models_loaded.append("tfidf")

            if self.graph:
                self.tfidf_reranker = GraphContextReranker(tfidf, self.graph)
                self.models_loaded.append("tfidf_graph")

        # 2. Fit Neural Models if available
        if neural_available and self.code_units:
            cache_dir = self.workspace.cache_dir
            try:
                sem = SemanticRetriever(cache_dir=cache_dir, include_metadata=True)
                sem.fit(self.code_units)
                self.semantic_retriever = sem
                self.models_loaded.append("semantic")

                if self.graph:
                    self.semantic_reranker = GraphContextReranker(sem, self.graph)
                    self.models_loaded.append("semantic_graph")
            except Exception as e:
                pass

            try:
                code_t = CodeTransformerRetriever(cache_dir=cache_dir, include_metadata=True)
                code_t.fit(self.code_units)
                self.code_retriever = code_t
                self.models_loaded.append("code")

                if self.graph:
                    self.code_reranker = GraphContextReranker(code_t, self.graph)
                    self.models_loaded.append("code_graph")
            except Exception as e:
                pass

            # 3. Fit Hybrid RRF
            if self.tfidf_retriever and self.code_retriever:
                try:
                    self.hybrid_retriever = HybridRetriever(
                        retrievers={"tfidf": self.tfidf_retriever, "code": self.code_retriever},
                        k=60,
                        graph=self.graph,
                        candidate_depth=25,
                    )
                    self.models_loaded.append("hybrid")
                    if self.graph:
                        self.models_loaded.append("hybrid_graph")
                except Exception as e:
                    pass

        self.is_loaded = True


# Global Application State Holder managing multi-workspace containers
class EngineState:
    """Manages workspace manager and dynamically caches fitted workspace containers."""

    def __init__(self) -> None:
        self.workspace_manager = WorkspaceManager(DATA_DIR)
        self.containers: Dict[str, WorkspaceContainer] = {}
        self.neural_available: bool = False
        self._lock = threading.Lock()

    def get_container(self, workspace_id: str = "default") -> WorkspaceContainer:
        """Get or lazily load container for specified workspace with thread safety."""
        clean_id = workspace_id.strip() if workspace_id else "default"
        with self._lock:
            ws = self.workspace_manager.get_workspace(clean_id)
            if ws.id not in self.containers:
                container = WorkspaceContainer(ws)
                container.load(neural_available=self.neural_available)
                self.containers[ws.id] = container
            return self.containers[ws.id]

    def invalidate_container(self, workspace_id: str) -> None:
        """Evict cached container to reload newly indexed units/models with thread safety."""
        clean_id = workspace_id.strip().lower()
        with self._lock:
            self.containers.pop(clean_id, None)
            self.containers.pop(workspace_id, None)

    # Backward-compatible property delegates for canonical 'default' workspace
    @property
    def code_units(self) -> List[Dict[str, Any]]:
        return self.get_container("default").code_units

    @property
    def code_units_by_id(self) -> Dict[str, Dict[str, Any]]:
        return self.get_container("default").code_units_by_id

    @property
    def graph(self) -> Optional[RepositoryGraph]:
        return self.get_container("default").graph

    @property
    def tfidf_retriever(self) -> Optional[TfidfRetriever]:
        return self.get_container("default").tfidf_retriever

    @property
    def semantic_retriever(self) -> Optional[SemanticRetriever]:
        return self.get_container("default").semantic_retriever

    @property
    def code_retriever(self) -> Optional[CodeTransformerRetriever]:
        return self.get_container("default").code_retriever

    @property
    def hybrid_retriever(self) -> Optional[HybridRetriever]:
        return self.get_container("default").hybrid_retriever

    @property
    def tfidf_reranker(self) -> Optional[GraphContextReranker]:
        return self.get_container("default").tfidf_reranker

    @property
    def semantic_reranker(self) -> Optional[GraphContextReranker]:
        return self.get_container("default").semantic_reranker

    @property
    def code_reranker(self) -> Optional[GraphContextReranker]:
        return self.get_container("default").code_reranker

    @property
    def models_loaded(self) -> List[str]:
        return self.get_container("default").models_loaded


state = EngineState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup handler: checks neural dependencies and pre-warms default workspace."""
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
        state.neural_available = True
    except ImportError:
        state.neural_available = False
        print("[Warning] sentence-transformers or torch is missing in current Python. Neural models disabled.")

    # Pre-warm default research workspace
    state.get_container("default")
    yield


# Create FastAPI application
app = FastAPI(
    title="CodeSage Search & Evaluation API",
    description="Backend API for CodeSage multi-repository code intelligence, workspace management, and empirical evaluation.",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS configuration for local web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Workspace Management Endpoints
# ==============================================================================

@app.get("/api/workspaces", summary="List Registered Workspaces")
def list_workspaces() -> List[Dict[str, Any]]:
    """List all registered workspaces with masked host paths for privacy and security."""
    return [ws.to_dict(mask_absolute_path=True) for ws in state.workspace_manager.list_workspaces()]


@app.post("/api/workspaces", summary="Register New Workspace", status_code=status.HTTP_201_CREATED)
def register_workspace(req: RegisterWorkspaceRequest) -> Dict[str, Any]:
    """Register a new permitted local repository workspace without modifying target sources."""
    try:
        ws = state.workspace_manager.register_workspace(req.id, req.repo_path, req.name)
        return ws.metadata.to_dict(mask_absolute_path=True)
    except (InvalidWorkspaceIdError, UnsafePathError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/workspaces/github", summary="Register & Ingest GitHub Workspace", status_code=status.HTTP_201_CREATED)
def register_github_workspace_endpoint(req: RegisterGithubWorkspaceRequest) -> Dict[str, Any]:
    """Register and safely ingest a public GitHub repository with snapshot isolation."""
    try:
        ws = state.workspace_manager.register_github_workspace(req.url, name=req.name)
        if req.auto_index:
            ws = state.workspace_manager.fetch_and_index_github_workspace(
                ws.id,
                max_files=req.max_files,
                max_file_size_kb=req.max_file_size_kb,
            )
            state.invalidate_container(ws.id)

        resp = ws.metadata.to_dict(mask_absolute_path=True)
        return resp
    except InvalidGithubUrlError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DuplicateWorkspaceError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except OversizedRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except UnsafeRepositoryContentError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except GithubFetchError as e:
        err_msg = str(e)
        if "timed out" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=sanitize_error_message(err_msg))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=sanitize_error_message(err_msg))
    except (WorkspaceError, GithubIngestionError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_error_message(str(e)))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=sanitize_error_message(str(e)))


@app.get("/api/workspaces/{workspace_id}", summary="Get Workspace Details")
def get_workspace_details(workspace_id: str) -> Dict[str, Any]:
    """Get metadata, index counts, and status for a specific workspace."""
    try:
        ws = state.workspace_manager.get_workspace(workspace_id)
        return ws.metadata.to_dict(mask_absolute_path=True)
    except WorkspaceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except WorkspaceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_error_message(str(e)))


@app.post("/api/workspaces/{workspace_id}/index", summary="Index Workspace")
def index_workspace_endpoint(
    workspace_id: str,
    req: Optional[IndexWorkspaceRequest] = None,
) -> Dict[str, Any]:
    """Trigger indexing or incremental re-indexing for a workspace."""
    try:
        incremental = req.incremental if req else False
        max_files = req.max_files if req else None
        max_file_size_kb = req.max_file_size_kb if req else 512
        max_unit_lines = req.max_unit_lines if req else 250

        if incremental:
            ws, stats = state.workspace_manager.incremental_index(
                workspace_id,
                max_file_size_kb=max_file_size_kb,
                max_unit_lines=max_unit_lines,
            )
        else:
            ws = state.workspace_manager.index_workspace(
                workspace_id,
                max_files=max_files,
                max_file_size_kb=max_file_size_kb,
                max_unit_lines=max_unit_lines,
            )
            stats = {"mode": "full", "unit_count": ws.metadata.unit_count}

        state.invalidate_container(workspace_id)
        resp = ws.metadata.to_dict(mask_absolute_path=True)
        resp["index_stats"] = stats
        return resp
    except WorkspaceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except WorkspaceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_error_message(str(e)))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indexing failed: {sanitize_error_message(str(e))}")


@app.post("/api/workspaces/{workspace_id}/refresh", summary="Refresh GitHub Workspace")
def refresh_github_workspace_endpoint(
    workspace_id: str,
    req: Optional[IndexWorkspaceRequest] = None,
) -> Dict[str, Any]:
    """Check remote GitHub repository for new commits and atomically refresh source and index."""
    try:
        max_files = req.max_files if req else None
        max_file_size_kb = req.max_file_size_kb if req else 512
        max_unit_lines = req.max_unit_lines if req else 250

        ws, res = state.workspace_manager.refresh_github_workspace(
            workspace_id,
            max_files=max_files,
            max_file_size_kb=max_file_size_kb,
            max_unit_lines=max_unit_lines,
        )
        if res.get("status") == "updated":
            state.invalidate_container(workspace_id)
        return res
    except WorkspaceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except OversizedRepositoryError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except UnsafeRepositoryContentError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except GithubFetchError as e:
        err_msg = str(e)
        if "timed out" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=sanitize_error_message(err_msg))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=sanitize_error_message(err_msg))
    except (WorkspaceError, GithubIngestionError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_error_message(str(e)))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=sanitize_error_message(str(e)))


@app.delete("/api/workspaces/{workspace_id}", summary="Delete Workspace Index")
def delete_workspace_endpoint(workspace_id: str) -> Dict[str, Any]:
    """Remove a workspace's generated index artifacts without touching target source files."""
    try:
        state.workspace_manager.delete_workspace(workspace_id)
        state.invalidate_container(workspace_id)
        return {
            "status": "deleted",
            "id": workspace_id,
            "message": "Workspace index artifacts deleted. Source files remain untouched.",
        }
    except WorkspaceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace '{workspace_id}' not found.")
    except WorkspaceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# ==============================================================================
# Search, Retrieval & Health Endpoints
# ==============================================================================

@app.get("/api/health", summary="System Health & Index Status")
def get_health() -> Dict[str, Any]:
    """Return health status, index counts, graph nodes, and loaded models."""
    default_ws = state.get_container("default")
    workspaces = state.workspace_manager.list_workspaces()
    return {
        "status": "online",
        "workspace": "DocuCharm",
        "active_workspace": "default",
        "workspaces": [ws.to_dict(mask_absolute_path=True) for ws in workspaces],
        "indexed_units": len(default_ws.code_units),
        "graph_nodes": len(default_ws.graph.nodes) if default_ws.graph else 0,
        "graph_edges": len(default_ws.graph.edges) if default_ws.graph else 0,
        "graph_unresolved_refs": len(default_ws.graph.unresolved_references) if default_ws.graph else 0,
        "models_loaded": default_ws.models_loaded,
        "index_file": str(INDEX_PATH.name),
    }


@app.get("/api/models", summary="List Available Retrieval Models")
def get_models(workspace_id: str = Query(default="default")) -> List[ModelInfo]:
    """Return list of supported retrieval engines and specifications for active workspace."""
    try:
        container = state.get_container(workspace_id)
        loaded = container.models_loaded
    except Exception:
        loaded = state.models_loaded

    models = [
        ModelInfo(
            id="tfidf",
            name="TF-IDF (Lexical Baseline)",
            category="lexical",
            description="scikit-learn sublinear TF-IDF with code-aware subword tokenization and metadata headers.",
            dimensions="Sparse vocabulary (~1,200 tokens)",
            parameters="0",
            weight_mb=0.0,
            supports_graph_reranking=True,
            is_available="tfidf" in loaded,
        ),
        ModelInfo(
            id="semantic",
            name="General-Purpose Semantic (Sentence-Transformers)",
            category="dense_sentence",
            description="Compact 6-layer MiniLM (all-MiniLM-L6-v2) for natural-language intent and short descriptions.",
            dimensions="384 dimensions",
            parameters="22.7M",
            weight_mb=86.7,
            supports_graph_reranking=True,
            is_available="semantic" in loaded,
        ),
        ModelInfo(
            id="code",
            name="Code-Specific Transformer (CodeSearchNet)",
            category="dense_code",
            description="Pretrained 6-layer DistilRoBERTa bi-encoder (st-codesearch-distilroberta-base) with 512-token window.",
            dimensions="768 dimensions",
            parameters="82.1M",
            weight_mb=313.3,
            supports_graph_reranking=True,
            is_available="code" in loaded,
        ),
        ModelInfo(
            id="tfidf_graph",
            name="TF-IDF + Graph Context Reranker",
            category="hybrid_graph",
            description="TF-IDF lexical retrieval combined with graph neighborhood affinity reranking.",
            dimensions="Sparse + Graph Adjacency",
            parameters="0",
            weight_mb=0.0,
            supports_graph_reranking=False,
            is_available="tfidf_graph" in loaded,
        ),
        ModelInfo(
            id="semantic_graph",
            name="Semantic + Graph Context Reranker",
            category="hybrid_graph",
            description="Dense general embeddings combined with syntax relationship graph neighborhood affinity.",
            dimensions="384 dimensions + Graph",
            parameters="22.7M",
            weight_mb=86.7,
            supports_graph_reranking=False,
            is_available="semantic_graph" in loaded,
        ),
        ModelInfo(
            id="code_graph",
            name="Code Transformer + Graph Context Reranker",
            category="hybrid_graph",
            description="Dense code representation combined with repository syntax relationship graph neighborhood affinity.",
            dimensions="768 dimensions + Graph",
            parameters="82.1M",
            weight_mb=313.3,
            supports_graph_reranking=False,
            is_available="code_graph" in loaded,
        ),
        ModelInfo(
            id="hybrid",
            name="Hybrid RRF (TF-IDF + Code Transformer)",
            category="hybrid",
            description="Reciprocal Rank Fusion (k=60) combining lexical subword precision with dense CodeSearchNet semantics.",
            dimensions="Sparse + 768 dimensions",
            parameters="82.1M",
            weight_mb=313.3,
            supports_graph_reranking=True,
            is_available="hybrid" in loaded,
        ),
        ModelInfo(
            id="hybrid_graph",
            name="Hybrid RRF + Graph Context Reranker",
            category="hybrid_graph",
            description="RRF fused candidates (TF-IDF + Code Transformer) reranked using repository relationship graph neighborhood affinity.",
            dimensions="Sparse + 768 dimensions + Graph",
            parameters="82.1M",
            weight_mb=313.3,
            supports_graph_reranking=False,
            is_available="hybrid_graph" in loaded,
        ),
    ]
    return models


class _HybridWrapper:
    """Wrapper adapting HybridRetriever to standard search(query, top_k) interface."""

    def __init__(self, hybrid: HybridRetriever, rerank: bool = False):
        self.hybrid = hybrid
        self.rerank = rerank
        self.unit_ids = hybrid.unit_ids

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        return self.hybrid.search(query, top_k=top_k, rerank=self.rerank)


def _resolve_retriever(container: WorkspaceContainer, model_id: str, rerank: bool = False):
    """Resolve retriever instance for a given workspace container based on model_id and rerank flag."""
    if model_id == "tfidf":
        return container.tfidf_reranker if rerank and container.tfidf_reranker else container.tfidf_retriever
    elif model_id == "tfidf_graph":
        return container.tfidf_reranker or container.tfidf_retriever
    elif model_id == "semantic":
        return container.semantic_reranker if rerank and container.semantic_reranker else container.semantic_retriever
    elif model_id == "semantic_graph":
        return container.semantic_reranker or container.semantic_retriever
    elif model_id == "code":
        return container.code_reranker if rerank and container.code_reranker else container.code_retriever
    elif model_id == "code_graph":
        return container.code_reranker or container.code_retriever
    elif model_id in ["hybrid", "rrf"]:
        if not container.hybrid_retriever:
            return None
        should_rerank = rerank or False
        return _HybridWrapper(container.hybrid_retriever, rerank=should_rerank)
    elif model_id == "hybrid_graph":
        if not container.hybrid_retriever:
            return None
        return _HybridWrapper(container.hybrid_retriever, rerank=True)
    return None


@app.post("/api/search", summary="Execute Code Search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Execute code search using requested model and workspace with input validation and bounded results."""
    clean_query = request.query.strip()
    if not clean_query:
        return SearchResponse(
            query=request.query,
            model=request.model,
            top_k=request.top_k,
            rerank_enabled=request.rerank,
            latency_ms=0.0,
            total_results=0,
            workspace_id=request.workspace_id,
            results=[],
        )

    try:
        container = state.get_container(request.workspace_id)
    except WorkspaceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{request.workspace_id}' not found.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error accessing workspace '{request.workspace_id}': {e}",
        )

    if not container.code_units:
        return SearchResponse(
            query=clean_query,
            model=request.model,
            top_k=request.top_k,
            rerank_enabled=request.rerank,
            latency_ms=0.0,
            total_results=0,
            workspace_id=request.workspace_id,
            results=[],
        )

    retriever = _resolve_retriever(container, request.model, request.rerank)
    if retriever is None:
        if request.model in ["semantic", "semantic_graph", "code", "code_graph", "hybrid", "hybrid_graph"] and request.model not in container.models_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Requested model '{request.model}' is unavailable because neural dependencies (sentence-transformers / torch) or index are not loaded.",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested model '{request.model}' is not available or not loaded.",
        )

    t0 = time.perf_counter()
    try:
        raw_results: List[SearchResult] = retriever.search(clean_query, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retriever execution failed for model '{request.model}': {str(e)}",
        )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    formatted_items = [
        SearchResultItem(
            unit_id=r.unit_id,
            score=round(float(r.score), 4),
            rel_path=r.rel_path,
            language=r.language,
            symbol_name=r.symbol_name,
            unit_type=r.unit_type,
            start_line=r.start_line,
            end_line=r.end_line,
            extraction_method=r.extraction_method,
            preview=r.preview,
            source_code=r.source_code,
            model_ranks=getattr(r, "model_ranks", None),
            rrf_score=getattr(r, "rrf_score", None),
            graph_boost=getattr(r, "graph_boost", None),
            graph_score=getattr(r, "graph_score", None),
        )
        for r in raw_results
    ]

    return SearchResponse(
        query=clean_query,
        model=request.model,
        top_k=request.top_k,
        rerank_enabled=request.rerank or "_graph" in request.model,
        latency_ms=elapsed_ms,
        total_results=len(formatted_items),
        workspace_id=request.workspace_id,
        results=formatted_items,
    )


@app.post("/api/compare", summary="Multi-Engine Comparison", response_model=CompareResponse)
def compare_search(request: CompareRequest) -> CompareResponse:
    """Execute multi-engine comparison across TF-IDF, Semantic, and Code Transformer side-by-side."""
    clean_query = request.query.strip()
    if not clean_query:
        return CompareResponse(
            query=request.query,
            top_k=request.top_k,
            total_latency_ms=0.0,
            workspace_id=request.workspace_id,
            systems={},
        )

    try:
        container = state.get_container(request.workspace_id)
    except WorkspaceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{request.workspace_id}' not found.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error accessing workspace '{request.workspace_id}': {e}",
        )

    configs = [
        ("tfidf", "TF-IDF (Lexical)", container.tfidf_retriever),
        ("semantic", "General Semantic (all-MiniLM)", container.semantic_retriever),
        ("code", "Code Transformer (st-codesearch)", container.code_retriever),
    ]

    if container.hybrid_retriever:
        configs.append(("hybrid", "Hybrid RRF (TF-IDF + Code)", _HybridWrapper(container.hybrid_retriever, rerank=False)))

    if request.with_graph and container.graph:
        configs.extend([
            ("tfidf_graph", "TF-IDF + Graph Rerank", container.tfidf_reranker),
            ("semantic_graph", "Semantic + Graph Rerank", container.semantic_reranker),
            ("code_graph", "Code Transformer + Graph Rerank", container.code_reranker),
        ])
        if container.hybrid_retriever:
            configs.append(("hybrid_graph", "Hybrid RRF + Graph Rerank", _HybridWrapper(container.hybrid_retriever, rerank=True)))

    systems_out: Dict[str, CompareSystemResult] = {}
    t_start = time.perf_counter()

    for slug, display_name, retriever in configs:
        if retriever is None:
            continue
        q_start = time.perf_counter()
        try:
            raw_res: List[SearchResult] = retriever.search(clean_query, top_k=request.top_k)
            q_latency = round((time.perf_counter() - q_start) * 1000.0, 2)

            items = [
                SearchResultItem(
                    unit_id=r.unit_id,
                    score=round(float(r.score), 4),
                    rel_path=r.rel_path,
                    language=r.language,
                    symbol_name=r.symbol_name,
                    unit_type=r.unit_type,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    extraction_method=r.extraction_method,
                    preview=r.preview,
                    source_code=r.source_code,
                    model_ranks=getattr(r, "model_ranks", None),
                    rrf_score=getattr(r, "rrf_score", None),
                    graph_boost=getattr(r, "graph_boost", None),
                    graph_score=getattr(r, "graph_score", None),
                )
                for r in raw_res
            ]
            systems_out[slug] = CompareSystemResult(
                name=display_name,
                model_slug=slug,
                latency_ms=q_latency,
                results=items,
            )
        except Exception:
            continue

    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return CompareResponse(
        query=clean_query,
        top_k=request.top_k,
        total_latency_ms=total_time_ms,
        workspace_id=request.workspace_id,
        systems=systems_out,
    )


@app.get("/api/evaluations", summary="Empirical Benchmark & Study Metrics")
def get_evaluations() -> Dict[str, Any]:
    """Serve verified, recorded empirical evaluation reports.
    
    Reads directly from persisted evaluation files. Does not generate fictional data.
    """
    out: Dict[str, Any] = {}

    # 1. 3-Way Benchmark
    if BENCHMARK_JSON_PATH.exists():
        try:
            with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
                out["benchmark_comparison"] = json.load(f)
        except Exception:
            out["benchmark_comparison"] = None
    else:
        out["benchmark_comparison"] = None

    # 2. Milestone 5 Research Study (Ablations, Bootstrap CIs, Splits)
    if STUDY_JSON_PATH.exists():
        try:
            with open(STUDY_JSON_PATH, "r", encoding="utf-8") as f:
                out["study_results"] = json.load(f)
        except Exception:
            out["study_results"] = None
    else:
        out["study_results"] = None

    # 3. Graph Reranking Experiment Results
    if GRAPH_RERANK_JSON_PATH.exists():
        try:
            with open(GRAPH_RERANK_JSON_PATH, "r", encoding="utf-8") as f:
                out["graph_reranking_results"] = json.load(f)
        except Exception:
            out["graph_reranking_results"] = None
    else:
        out["graph_reranking_results"] = None

    # 4. Hybrid Reciprocal Rank Fusion Results
    if HYBRID_JSON_PATH.exists():
        try:
            with open(HYBRID_JSON_PATH, "r", encoding="utf-8") as f:
                out["hybrid_fusion_results"] = json.load(f)
        except Exception:
            out["hybrid_fusion_results"] = None
    else:
        out["hybrid_fusion_results"] = None

    return out


@app.get("/api/units/{unit_id:path}", summary="Get Code Unit Details")
def get_code_unit(
    unit_id: str,
    workspace_id: str = Query(default="default"),
) -> Dict[str, Any]:
    """Retrieve details and graph connections for a specific code unit in the target workspace."""
    clean_id = unit_id.strip()

    try:
        container = state.get_container(workspace_id)
    except WorkspaceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )
    except WorkspaceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitize_error_message(str(e)),
        )

    unit = container.code_units_by_id.get(clean_id)
    if not unit:
        unit = container.code_units_by_id.get(clean_id.lstrip("/"))
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Code unit '{unit_id}' not found in workspace '{workspace_id}'.",
        )

    # Gather graph context if available
    graph_context: Dict[str, Any] = {"incoming": [], "outgoing": []}
    if container.graph:
        node_id = f"unit:{unit_id}"
        out_neighbors = container.graph.get_neighbors(node_id, direction="outgoing")
        in_neighbors = container.graph.get_neighbors(node_id, direction="incoming")
        graph_context["outgoing"] = [
            {"target": tgt, "relationship": rel, "confidence": conf}
            for tgt, rel, conf in out_neighbors
        ]
        graph_context["incoming"] = [
            {"source": src, "relationship": rel, "confidence": conf}
            for src, rel, conf in in_neighbors
        ]

    return {
        "unit": unit,
        "workspace_id": workspace_id,
        "graph_context": graph_context,
    }
