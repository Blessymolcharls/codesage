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
from codesage.retrieval import TfidfRetriever, SearchResult
from codesage.semantic import SemanticRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeSage: Search indexed code units using lexical TF-IDF, general semantic, or code Transformer."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="",
        help="Natural-language or code-token search query (e.g., 'validate uploaded PDF')",
    )
    parser.add_argument(
        "--mode",
        choices=["tfidf", "semantic", "code"],
        default="tfidf",
        help="Retrieval engine to use: 'tfidf' (lexical), 'semantic' (all-MiniLM-L6-v2), or 'code' (st-codesearch-distilroberta-base) (default: 'tfidf')",
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
) -> None:
    if mode == "code":
        mode_title = "Code-Specific Transformer Retrieval (st-codesearch-distilroberta-base)"
    elif mode == "semantic":
        mode_title = "General-Purpose Semantic Retrieval (all-MiniLM-L6-v2)"
    else:
        mode_title = "TF-IDF Lexical Retrieval"

    print(f"\n{'=' * 80}")
    print(f"CodeSage Search Results — [{mode_title}]")
    print(f"Query: '{query}'")
    print(f"Index: {total_units} units indexed | Top results requested: {len(results)} | Latency: {latency_ms:.2f} ms")
    print(f"{'=' * 80}\n")

    if not results:
        print("No matching code units found.\n")
        return

    for idx, r in enumerate(results, start=1):
        symbol_str = f" :: {r.symbol_name}" if r.symbol_name else ""
        print(f"[{idx}] Score: {r.score:.4f} | {r.rel_path}:{r.start_line}-{r.end_line}{symbol_str}")
        print(f"    Unit ID   : {r.unit_id}")
        print(f"    Language  : {r.language} | Type: {r.unit_type} | Extractor: {r.extraction_method}")
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

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: Index file not found at '{index_path}'.", file=sys.stderr)
        print("Please run `python scripts/build_index.py` first to generate the index.", file=sys.stderr)
        sys.exit(1)

    cache_dir = project_root / "data" / "cache"

    if args.mode == "code":
        retriever = CodeTransformerRetriever(
            cache_dir=cache_dir,
            force_recompute=args.force_recompute,
        )
    elif args.mode == "semantic":
        retriever = SemanticRetriever(
            cache_dir=cache_dir,
            force_recompute=args.force_recompute,
        )
    else:
        retriever = TfidfRetriever(use_char_ngrams=args.char_ngrams)

    try:
        retriever.fit_from_jsonl(index_path)
    except Exception as e:
        print(f"Error loading index / model: {e}", file=sys.stderr)
        sys.exit(1)

    start_t = time.perf_counter()
    results = retriever.search(query, top_k=args.top_k)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    if args.json_output:
        out = {
            "query": query,
            "mode": args.mode,
            "total_indexed_units": len(retriever),
            "top_k": args.top_k,
            "latency_ms": round(latency_ms, 3),
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(out, indent=2))
    else:
        format_terminal_results(results, query, len(retriever), args.mode, latency_ms)


if __name__ == "__main__":
    main()
