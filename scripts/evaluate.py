#!/usr/bin/env python3
"""CLI entry point for CodeSage benchmark evaluation (TF-IDF vs Semantic vs Code Transformer)."""

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
from codesage.evaluation import (
    ThreeWayBenchmarkResult,
    evaluate_dataset,
    export_aggregate_json,
    export_query_results_csv,
    export_three_way_csv,
    load_evaluation_queries,
    run_paired_significance_test,
)
from codesage.retrieval import TfidfRetriever
from codesage.semantic import SemanticRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeSage: Benchmark IR evaluation comparing TF-IDF, general semantic, and code Transformer."
    )
    parser.add_argument(
        "--mode",
        choices=["tfidf", "semantic", "code", "compare"],
        default="compare",
        help="Evaluation mode: 'tfidf', 'semantic', 'code', or 'compare' (all 3 models) (default: 'compare')",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=str(project_root / "data" / "evaluation_queries.jsonl"),
        help="Path to evaluation queries JSONL (default: 'data/evaluation_queries.jsonl')",
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
        help="Top-K retrieval depth for evaluation (default: 10)",
    )
    parser.add_argument(
        "--char-ngrams",
        action="store_true",
        help="[TF-IDF only] Enable character subword n-grams in TF-IDF indexer",
    )
    parser.add_argument(
        "--split",
        choices=["all", "dev", "test"],
        default="all",
        help="Evaluation split filter: 'all', 'dev', or 'test' (default: 'all')",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Ablation: exclude symbol names and file paths from document representation",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="[Neural models] Bypass local embedding cache and re-encode all units",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional file path to export complete evaluation metrics JSON report",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Optional file path to export query-level metric results CSV",
    )
    return parser.parse_args()


def get_all_models_storage_info(cache_dir: Path) -> dict:
    """Calculate disk footprint for model weights and embedding caches across all models."""
    cache_sizes: dict[str, float] = {}
    if cache_dir.exists():
        for f in cache_dir.glob("*.npz"):
            cache_sizes[f.name] = round(f.stat().st_size / 1024.0, 2)

    return {
        "models": {
            "tfidf": {
                "name": "scikit-learn TfidfVectorizer",
                "parameters": "0",
                "embedding_dim": "Sparse vocabulary",
                "model_weight_mb": 0.0,
            },
            "semantic": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "parameters": "22.7M",
                "embedding_dim": 384,
                "model_weight_mb": 86.7,
            },
            "code_transformer": {
                "name": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
                "parameters": "82.1M",
                "embedding_dim": 768,
                "model_weight_mb": 313.3,
            },
        },
        "embedding_caches_kb": cache_sizes,
    }


def main() -> None:
    args = parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: Evaluation queries file not found at '{queries_path}'.", file=sys.stderr)
        sys.exit(1)

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: Index file not found at '{index_path}'.", file=sys.stderr)
        sys.exit(1)

    queries = load_evaluation_queries(queries_path)
    cache_dir = project_root / "data" / "cache"

    include_metadata = not args.no_metadata
    split_filter = args.split

    if args.mode == "tfidf":
        print(f"Evaluating TF-IDF baseline (split: {split_filter}, metadata: {include_metadata}) on {len(queries)} queries...")
        retriever = TfidfRetriever(use_char_ngrams=args.char_ngrams, include_metadata=include_metadata)
        retriever.fit_from_jsonl(index_path)
        query_results, aggregate = evaluate_dataset(retriever, queries, top_k=args.top_k, split_filter=split_filter)
        print(aggregate.summary_table())

        if args.export_json:
            export_aggregate_json(aggregate, args.export_json)
            print(f"Exported metrics JSON to: {args.export_json}")
        if args.export_csv:
            export_query_results_csv(query_results, args.export_csv)
            print(f"Exported query results CSV to: {args.export_csv}")

    elif args.mode == "semantic":
        print(f"Evaluating Semantic baseline (all-MiniLM-L6-v2) (split: {split_filter}, metadata: {include_metadata}) on {len(queries)} queries...")
        retriever = SemanticRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=include_metadata)
        retriever.fit_from_jsonl(index_path)
        query_results, aggregate = evaluate_dataset(retriever, queries, top_k=args.top_k, split_filter=split_filter)
        print(aggregate.summary_table())

        if args.export_json:
            export_aggregate_json(aggregate, args.export_json)
            print(f"Exported metrics JSON to: {args.export_json}")
        if args.export_csv:
            export_query_results_csv(query_results, args.export_csv)
            print(f"Exported query results CSV to: {args.export_csv}")

    elif args.mode == "code":
        print(f"Evaluating Code-Specific Transformer (st-codesearch-distilroberta-base) (split: {split_filter}, metadata: {include_metadata}) on {len(queries)} queries...")
        retriever = CodeTransformerRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=include_metadata)
        retriever.fit_from_jsonl(index_path)
        query_results, aggregate = evaluate_dataset(retriever, queries, top_k=args.top_k, split_filter=split_filter)
        print(aggregate.summary_table())

        if args.export_json:
            export_aggregate_json(aggregate, args.export_json)
            print(f"Exported metrics JSON to: {args.export_json}")
        if args.export_csv:
            export_query_results_csv(query_results, args.export_csv)
            print(f"Exported query results CSV to: {args.export_csv}")

    else:  # compare mode (all 3 models)
        print(f"Running 3-Way Comparative Benchmark across {len(queries)} queries (split: {split_filter}, metadata: {include_metadata})...\n")

        # 1. TF-IDF
        print(">> [1/3] Benchmarking TF-IDF Lexical Engine...")
        t0 = time.perf_counter()
        tfidf_retriever = TfidfRetriever(use_char_ngrams=args.char_ngrams, include_metadata=include_metadata)
        tfidf_retriever.fit_from_jsonl(index_path)
        t_tfidf_index = time.perf_counter() - t0

        t_latencies = []
        for q in queries:
            q_start = time.perf_counter()
            _ = tfidf_retriever.search(q.query, top_k=args.top_k)
            t_latencies.append((time.perf_counter() - q_start) * 1000.0)

        tfidf_query_results, tfidf_aggregate = evaluate_dataset(tfidf_retriever, queries, top_k=args.top_k, split_filter=split_filter)

        # 2. General-Purpose Semantic
        print(">> [2/3] Benchmarking General Semantic Engine (all-MiniLM-L6-v2)...")
        t1 = time.perf_counter()
        semantic_retriever = SemanticRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=include_metadata)
        semantic_retriever.fit_from_jsonl(index_path)
        t_semantic_index = time.perf_counter() - t1

        s_latencies = []
        for q in queries:
            q_start = time.perf_counter()
            _ = semantic_retriever.search(q.query, top_k=args.top_k)
            s_latencies.append((time.perf_counter() - q_start) * 1000.0)

        semantic_query_results, semantic_aggregate = evaluate_dataset(semantic_retriever, queries, top_k=args.top_k, split_filter=split_filter)

        # 3. Code-Specific Transformer
        print(">> [3/3] Benchmarking Code-Specific Transformer (st-codesearch-distilroberta-base)...")
        t2 = time.perf_counter()
        code_retriever = CodeTransformerRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=include_metadata)
        code_retriever.fit_from_jsonl(index_path)
        t_code_index = time.perf_counter() - t2

        c_latencies = []
        for q in queries:
            q_start = time.perf_counter()
            _ = code_retriever.search(q.query, top_k=args.top_k)
            c_latencies.append((time.perf_counter() - q_start) * 1000.0)

        code_query_results, code_aggregate = evaluate_dataset(code_retriever, queries, top_k=args.top_k, split_filter=split_filter)

        # 4. Significance Testing
        paired_code_vs_tfidf = run_paired_significance_test(tfidf_query_results, code_query_results)
        paired_code_vs_semantic = run_paired_significance_test(semantic_query_results, code_query_results)
        paired_semantic_vs_tfidf = run_paired_significance_test(tfidf_query_results, semantic_query_results)

        paired_significance = {
            "code_vs_tfidf": paired_code_vs_tfidf,
            "code_vs_semantic": paired_code_vs_semantic,
            "semantic_vs_tfidf": paired_semantic_vs_tfidf,
        }

        storage_info = get_all_models_storage_info(cache_dir)

        three_way_result = ThreeWayBenchmarkResult(
            tfidf_aggregate=tfidf_aggregate,
            semantic_aggregate=semantic_aggregate,
            code_aggregate=code_aggregate,
            tfidf_latencies_ms=t_latencies,
            semantic_latencies_ms=s_latencies,
            code_latencies_ms=c_latencies,
            tfidf_indexing_time_s=t_tfidf_index,
            semantic_indexing_time_s=t_semantic_index,
            code_indexing_time_s=t_code_index,
            paired_significance=paired_significance,
            model_storage_info=storage_info,
        )

        print(three_way_result.summary_table())

        if args.export_json:
            json_path = Path(args.export_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(three_way_result.to_dict(), f, indent=2)
            print(f"\nExported 3-way comparative JSON report to: {json_path}")

        if args.export_csv:
            csv_path = Path(args.export_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            export_three_way_csv(
                tfidf_results=tfidf_query_results,
                semantic_results=semantic_query_results,
                code_results=code_query_results,
                output_path=csv_path,
            )
            print(f"Exported 3-way comparative CSV report to: {csv_path}")


if __name__ == "__main__":
    main()
