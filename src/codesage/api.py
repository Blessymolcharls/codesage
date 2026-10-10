"""FastAPI backend server wrapping CodeSage code search and evaluation."""

import json
import os
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
from codesage.retrieval import SearchResult, TfidfRetriever, load_code_units_from_jsonl
from codesage.semantic import SemanticRetriever

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "code_units.jsonl"
GRAPH_PATH = DATA_DIR / "relationship_graph.json"
CACHE_DIR = DATA_DIR / "cache"
BENCHMARK_JSON_PATH = DATA_DIR / "benchmark_comparison.json"
STUDY_JSON_PATH = DATA_DIR / "experiments" / "study_results.json"
GRAPH_RERANK_JSON_PATH = DATA_DIR / "experiments" / "graph_reranking_results.json"


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
        description="Retriever model: 'tfidf', 'semantic', 'code', 'tfidf_graph', 'semantic_graph', 'code_graph'",
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


class SearchResponse(BaseModel):
    query: str
    model: str
    top_k: int
    rerank_enabled: bool
    latency_ms: float
    total_results: int
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


# Shared Global State Holder
class EngineState:
    def __init__(self) -> None:
        self.code_units: List[Dict[str, Any]] = []
        self.code_units_by_id: Dict[str, Dict[str, Any]] = {}
        self.graph: Optional[RepositoryGraph] = None

        self.tfidf_retriever: Optional[TfidfRetriever] = None
        self.semantic_retriever: Optional[SemanticRetriever] = None
        self.code_retriever: Optional[CodeTransformerRetriever] = None

        # Graph Rerankers
        self.tfidf_reranker: Optional[GraphContextReranker] = None
        self.semantic_reranker: Optional[GraphContextReranker] = None
        self.code_reranker: Optional[GraphContextReranker] = None

        self.models_loaded: List[str] = []


state = EngineState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup handler: loads index, graph, and fits retrievers."""
    # 1. Load code units
    if INDEX_PATH.exists():
        state.code_units = load_code_units_from_jsonl(INDEX_PATH)
        state.code_units_by_id = {u["unit_id"]: u for u in state.code_units}
    else:
        state.code_units = []

    # 2. Load relationship graph
    if GRAPH_PATH.exists():
        state.graph = RepositoryGraph.load_json(GRAPH_PATH)
    elif state.code_units:
        builder = RepositoryGraphBuilder(repo_root=PROJECT_ROOT.parent / "DocuCharm")
        state.graph = builder.build(state.code_units)

    # 3. Initialize TF-IDF
    if state.code_units:
        tfidf = TfidfRetriever(include_metadata=True)
        tfidf.fit(state.code_units)
        state.tfidf_retriever = tfidf
        state.models_loaded.append("tfidf")

        if state.graph:
            state.tfidf_reranker = GraphContextReranker(tfidf, state.graph)
            state.models_loaded.append("tfidf_graph")

    # 4. Initialize Semantic Retriever with local cache
    try:
        sem = SemanticRetriever(cache_dir=CACHE_DIR, include_metadata=True)
        if state.code_units:
            sem.fit(state.code_units)
        state.semantic_retriever = sem
        state.models_loaded.append("semantic")

        if state.graph:
            state.semantic_reranker = GraphContextReranker(sem, state.graph)
            state.models_loaded.append("semantic_graph")
    except Exception as e:
        print(f"[Warning] Failed to pre-warm semantic retriever: {e}")

    # 5. Initialize Code Transformer Retriever with local cache
    try:
        code_t = CodeTransformerRetriever(cache_dir=CACHE_DIR, include_metadata=True)
        if state.code_units:
            code_t.fit(state.code_units)
        state.code_retriever = code_t
        state.models_loaded.append("code")

        if state.graph:
            state.code_reranker = GraphContextReranker(code_t, state.graph)
            state.models_loaded.append("code_graph")
    except Exception as e:
        print(f"[Warning] Failed to pre-warm code transformer retriever: {e}")

    yield


# Create FastAPI application
app = FastAPI(
    title="CodeSage Search & Evaluation API",
    description="Backend API for CodeSage multi-regime code intelligence, repository search, and empirical evaluation.",
    version="1.0.0",
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


@app.get("/api/health", summary="System Health & Index Status")
def get_health() -> Dict[str, Any]:
    """Return health status, index counts, graph nodes, and loaded models."""
    return {
        "status": "online",
        "workspace": "DocuCharm",
        "indexed_units": len(state.code_units),
        "graph_nodes": len(state.graph.nodes) if state.graph else 0,
        "graph_edges": len(state.graph.edges) if state.graph else 0,
        "graph_unresolved_refs": len(state.graph.unresolved_references) if state.graph else 0,
        "models_loaded": state.models_loaded,
        "index_file": str(INDEX_PATH.name),
    }


@app.get("/api/models", summary="List Available Retrieval Models")
def get_models() -> List[ModelInfo]:
    """Return list of supported retrieval engines and specifications."""
    return [
        ModelInfo(
            id="tfidf",
            name="TF-IDF (Lexical Baseline)",
            category="lexical",
            description="scikit-learn sublinear TF-IDF with code-aware subword tokenization and metadata headers.",
            dimensions="Sparse vocabulary (~1,200 tokens)",
            parameters="0",
            weight_mb=0.0,
            supports_graph_reranking=True,
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
        ),
        ModelInfo(
            id="semantic_graph",
            name="Semantic + Graph Context Reranker",
            category="hybrid_graph",
            description="General-purpose embeddings reranked using repository relationship graph context.",
            dimensions="384 dimensions + Graph",
            parameters="22.7M",
            weight_mb=86.7,
            supports_graph_reranking=False,
        ),
        ModelInfo(
            id="code_graph",
            name="Code Transformer + Graph Context Reranker",
            category="hybrid_graph",
            description="CodeSearchNet embeddings reranked using repository relationship graph context.",
            dimensions="768 dimensions + Graph",
            parameters="82.1M",
            weight_mb=313.3,
            supports_graph_reranking=False,
        ),
    ]


def _resolve_retriever(model_id: str, rerank: bool = False):
    """Resolve retriever instance based on model_id and rerank flag."""
    if model_id == "tfidf":
        return state.tfidf_reranker if rerank and state.tfidf_reranker else state.tfidf_retriever
    elif model_id == "tfidf_graph":
        return state.tfidf_reranker or state.tfidf_retriever
    elif model_id == "semantic":
        return state.semantic_reranker if rerank and state.semantic_reranker else state.semantic_retriever
    elif model_id == "semantic_graph":
        return state.semantic_reranker or state.semantic_retriever
    elif model_id == "code":
        return state.code_reranker if rerank and state.code_reranker else state.code_retriever
    elif model_id == "code_graph":
        return state.code_reranker or state.code_retriever
    return None


@app.post("/api/search", summary="Execute Code Search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Execute code search using requested model with input validation and bounded results."""
    clean_query = request.query.strip()
    if not clean_query:
        return SearchResponse(
            query=request.query,
            model=request.model,
            top_k=request.top_k,
            rerank_enabled=request.rerank,
            latency_ms=0.0,
            total_results=0,
            results=[],
        )

    retriever = _resolve_retriever(request.model, request.rerank)
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested model '{request.model}' is not available or not loaded.",
        )

    t0 = time.perf_counter()
    raw_results: List[SearchResult] = retriever.search(clean_query, top_k=request.top_k)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Convert to bounded response items
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
        results=formatted_items,
    )


@app.post("/api/compare", summary="3-Way Multi-Engine Comparison", response_model=CompareResponse)
def compare_search(request: CompareRequest) -> CompareResponse:
    """Execute multi-engine comparison across TF-IDF, Semantic, and Code Transformer side-by-side."""
    clean_query = request.query.strip()
    if not clean_query:
        return CompareResponse(
            query=request.query,
            top_k=request.top_k,
            total_latency_ms=0.0,
            systems={},
        )

    configs = [
        ("tfidf", "TF-IDF (Lexical)", state.tfidf_retriever),
        ("semantic", "General Semantic (all-MiniLM)", state.semantic_retriever),
        ("code", "Code Transformer (st-codesearch)", state.code_retriever),
    ]

    if request.with_graph and state.graph:
        configs.extend([
            ("tfidf_graph", "TF-IDF + Graph Rerank", state.tfidf_reranker),
            ("semantic_graph", "Semantic + Graph Rerank", state.semantic_reranker),
            ("code_graph", "Code Transformer + Graph Rerank", state.code_reranker),
        ])

    systems_out: Dict[str, CompareSystemResult] = {}
    t_start = time.perf_counter()

    for slug, display_name, retriever in configs:
        if retriever is None:
            continue
        q_start = time.perf_counter()
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
            )
            for r in raw_res
        ]
        systems_out[slug] = CompareSystemResult(
            name=display_name,
            model_slug=slug,
            latency_ms=q_latency,
            results=items,
        )

    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return CompareResponse(
        query=clean_query,
        top_k=request.top_k,
        total_latency_ms=total_time_ms,
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

    return out


@app.get("/api/units/{unit_id:path}", summary="Get Code Unit Details")
def get_code_unit(unit_id: str) -> Dict[str, Any]:
    """Retrieve details and graph connections for a specific code unit."""
    clean_id = unit_id.strip()
    unit = state.code_units_by_id.get(clean_id)
    if not unit:
        # Also check without leading slash if any
        unit = state.code_units_by_id.get(clean_id.lstrip("/"))
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Code unit '{unit_id}' not found in index.",
        )

    # Gather graph context if available
    graph_context: Dict[str, Any] = {"incoming": [], "outgoing": []}
    if state.graph:
        node_id = f"unit:{unit_id}"
        out_neighbors = state.graph.get_neighbors(node_id, direction="outgoing")
        in_neighbors = state.graph.get_neighbors(node_id, direction="incoming")
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
        "graph_context": graph_context,
    }
