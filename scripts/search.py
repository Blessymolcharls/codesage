#!/usr/bin/env python3
"""CLI entry point for CodeSage code search (lexical TF-IDF, general semantic, and code Transformer)."""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure 'src' is in Python module search path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from codesage.code_semantic import CodeTransformerRetriever
from codesage.graph import RepositoryGraph
from codesage.graph_reranker import GraphContextReranker
from codesage.hybrid import HybridRetriever
from codesage.retrieval import SearchResult, TfidfRetriever
from codesage.semantic import SemanticRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeSage: Search indexed code units using lexical TF-IDF, semantic, code Transformer, or hybrid RRF."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="",
        help="Natural-language or code-token search query (e.g., 'validate uploaded PDF')",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Target workspace ID to search (e.g. 'docucharm' or a custom repo ID)",
    )
    parser.add_argument(
        "--mode",
        choices=["tfidf", "semantic", "code", "hybrid"],
        default="tfidf",
        help="Retrieval engine to use: 'tfidf', 'semantic', 'code', or 'hybrid' (RRF) (default: 'tfidf')",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply graph-context neighborhood affinity reranking using repository relationship graph",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=str(project_root / "data" / "code_units.jsonl"),
        help="Path to code units JSONL index file (default: 'data/code_units.jsonl')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked results to return (default: 10)",
    )
    parser.add_argument(
        "--char-ngrams",
        action="store_true",
        help="[TF-IDF only] Include subword character 3-5 ngrams in addition to word ngrams",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="[Neural only] Bypass local embedding cache and re-encode all units",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results in JSON format instead of terminal table",
    )
    return parser.parse_args()


def format_terminal_results(
    results: list[SearchResult],
    query: str,
    total_units: int,
    mode: str,
    latency_ms: float,
    workspace_id: str = "default",
    rerank_enabled: bool = False,
) -> None:
    if mode == "code":
        mode_title = "Code-Specific Transformer Retrieval (st-codesearch-distilroberta-base)"
    elif mode == "semantic":
        mode_title = "General-Purpose Semantic Retrieval (all-MiniLM-L6-v2)"
    elif mode == "hybrid":
        mode_title = "Hybrid RRF Retrieval (TF-IDF + CodeTransformer)"
    else:
        mode_title = "TF-IDF Lexical Retrieval"

    if rerank_enabled:
        mode_title += " + Graph Reranking"

    print(f"\n{'=' * 80}")
    print(f"CodeSage Search Results — [{mode_title}]")
    print(f"Workspace: {workspace_id} | Query: '{query}'")
    print(f"Index: {total_units} units indexed | Top results requested: {len(results)} | Latency: {latency_ms:.2f} ms")
    print(f"{'=' * 80}\n")

    if not results:
        print("No matching code units found.\n")
        return

    for idx, r in enumerate(results, start=1):
        symbol_str = f" :: {r.symbol_name}" if r.symbol_name else ""
        score_info = f"Score: {r.score:.4f}"
        if getattr(r, "rrf_score", 0.0):
            score_info = f"RRF Score: {getattr(r, 'rrf_score'):.5f}"
        if getattr(r, "graph_boost", 0.0):
            score_info += f" (+{getattr(r, 'graph_boost'):.4f} graph)"

        print(f"[{idx}] {score_info} | {r.rel_path}:{r.start_line}-{r.end_line}{symbol_str}")
        print(f"    Unit ID   : {r.unit_id}")
        print(f"    Language  : {r.language} | Type: {r.unit_type} | Extractor: {r.extraction_method}")
        if getattr(r, "model_ranks", None):
            print(f"    Fused Ranks: {r.model_ranks}")
        print(f"    Preview   :")
        for line in r.preview.splitlines():
            print(f"      {line}")
        print(f"{'-' * 80}")


def main() -> None:
    args = parse_args()

    query = args.query.strip()
    if not query:
        print("Error: Empty query provided. Please pass a search query.", file=sys.stderr)
        sys.exit(1)

    if args.workspace:
        from codesage.workspace import WorkspaceManager

        manager = WorkspaceManager(project_root / "data")
        try:
            ws = manager.get_workspace(args.workspace.strip())
            index_path = ws.units_path
            cache_dir = ws.cache_dir
            graph_path = ws.graph_path if ws.graph_path.exists() else None
            workspace_id = ws.id
        except Exception as e:
            print(f"Error resolving workspace '{args.workspace}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        index_path = Path(args.index)
        cache_dir = project_root / "data" / "cache"
        graph_path = project_root / "data" / "relationship_graph.json"
        workspace_id = "default"

    if not index_path.exists():
        print(f"Error: Index file not found at '{index_path}'.", file=sys.stderr)
        print("Please run `python scripts/build_index.py` first to generate the index.", file=sys.stderr)
        sys.exit(1)

    graph = None
    if args.rerank or args.mode == "hybrid":
        if graph_path and Path(graph_path).exists():
            try:
                graph = RepositoryGraph.load_json(Path(graph_path))
            except Exception as e:
                print(f"Warning: Failed to load graph from {graph_path}: {e}", file=sys.stderr)

    try:
        if args.mode == "hybrid":
            tfidf = TfidfRetriever(use_char_ngrams=args.char_ngrams, include_metadata=True)
            tfidf.fit_from_jsonl(index_path)

            code_retriever = CodeTransformerRetriever(
                cache_dir=cache_dir,
                force_recompute=args.force_recompute,
                include_metadata=True,
            )
            code_retriever.fit_from_jsonl(index_path)

            retriever = HybridRetriever(
                retrievers={"tfidf": tfidf, "code": code_retriever},
                graph=graph if args.rerank else None,
            )
            total_units = len(tfidf)
        elif args.mode == "code":
            retriever = CodeTransformerRetriever(
                cache_dir=cache_dir,
                force_recompute=args.force_recompute,
            )
            retriever.fit_from_jsonl(index_path)
            total_units = len(retriever)
            if args.rerank and graph:
                retriever = GraphContextReranker(retriever, graph)
        elif args.mode == "semantic":
            retriever = SemanticRetriever(
                cache_dir=cache_dir,
                force_recompute=args.force_recompute,
            )
            retriever.fit_from_jsonl(index_path)
            total_units = len(retriever)
            if args.rerank and graph:
                retriever = GraphContextReranker(retriever, graph)
        else:
            retriever = TfidfRetriever(use_char_ngrams=args.char_ngrams)
            retriever.fit_from_jsonl(index_path)
            total_units = len(retriever)
            if args.rerank and graph:
                retriever = GraphContextReranker(retriever, graph)
    except Exception as e:
        print(f"Error loading index / model: {e}", file=sys.stderr)
        sys.exit(1)

    start_t = time.perf_counter()
    if args.mode == "hybrid" and args.rerank and graph:
        results = retriever.search(query, top_k=args.top_k, rerank=True)
    else:
        results = retriever.search(query, top_k=args.top_k)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    rerank_effective = bool(args.rerank and graph is not None)

    if args.json_output:
        out = {
            "query": query,
            "mode": args.mode,
            "workspace_id": workspace_id,
            "rerank_enabled": rerank_effective,
            "total_indexed_units": total_units,
            "top_k": args.top_k,
            "latency_ms": round(latency_ms, 3),
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(out, indent=2))
    else:
        format_terminal_results(
            results,
            query,
            total_units,
            args.mode,
            latency_ms,
            workspace_id=workspace_id,
            rerank_enabled=rerank_effective,
        )


if __name__ == "__main__":
    main()
