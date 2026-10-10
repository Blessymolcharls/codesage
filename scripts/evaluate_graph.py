#!/usr/bin/env python3
"""Run graph-context reranking experiment against the three retrieval baselines."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure 'src' is in Python module search path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from codesage.code_semantic import CodeTransformerRetriever
from codesage.evaluation import (
    AggregateEvaluationResult,
    QueryEvaluationResult,
    evaluate_dataset,
    load_evaluation_queries,
)
from codesage.graph import RepositoryGraph, RepositoryGraphBuilder
from codesage.graph_reranker import GraphContextReranker
from codesage.retrieval import TfidfRetriever, load_code_units_from_jsonl
from codesage.semantic import SemanticRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeSage: Graph Context Reranking Experiment vs 3 Retrieval Baselines."
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
        "--graph",
        type=str,
        default=str(project_root / "data" / "relationship_graph.json"),
        help="Path to relationship graph JSON (default: 'data/relationship_graph.json')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K retrieval depth for evaluation (default: 10)",
    )
    parser.add_argument(
        "--split",
        choices=["all", "dev", "test"],
        default="all",
        help="Evaluation split filter: 'all', 'dev', or 'test' (default: 'all')",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Weight for base retrieval score (default: 0.75)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="Weight for graph context score (default: 0.25)",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=str(project_root / "data" / "experiments" / "graph_reranking_results.json"),
        help="Path to export evaluation metrics JSON (default: 'data/experiments/graph_reranking_results.json')",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=str(project_root / "data" / "experiments" / "graph_reranking_comparison.csv"),
        help="Path to export comparative CSV (default: 'data/experiments/graph_reranking_comparison.csv')",
    )
    return parser.parse_args()


def benchmark_system(
    system_name: str,
    retriever: Any,
    queries: List[Any],
    top_k: int,
    split_filter: str,
) -> Tuple[List[QueryEvaluationResult], AggregateEvaluationResult, List[float]]:
    """Evaluate a retriever system and record per-query latencies."""
    print(f"  Evaluating {system_name}...")
    latencies: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        _ = retriever.search(q.query, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    results, aggregate = evaluate_dataset(retriever, queries, top_k=top_k, split_filter=split_filter)
    return results, aggregate, latencies


def format_delta(delta: float, pct: bool = False) -> str:
    """Format numeric difference with explicit +/- sign."""
    if pct:
        return f"{delta:+.1f}%" if delta != 0 else "0.0%"
    return f"{delta:+.4f}" if delta != 0 else "0.0000"


def main() -> None:
    args = parse_args()
    queries_path = Path(args.queries)
    index_path = Path(args.index)
    graph_path = Path(args.graph)

    if not queries_path.exists():
        print(f"Error: Evaluation queries not found at '{queries_path}'", file=sys.stderr)
        sys.exit(1)
    if not index_path.exists():
        print(f"Error: Index file not found at '{index_path}'", file=sys.stderr)
        sys.exit(1)

    queries = load_evaluation_queries(queries_path)
    cache_dir = project_root / "data" / "cache"

    # Load or build graph
    if graph_path.exists():
        print(f"Loading repository graph from: {graph_path}")
        graph = RepositoryGraph.load_json(graph_path)
    else:
        print(f"Graph file not found; building from {index_path}...")
        units = load_code_units_from_jsonl(index_path)
        builder = RepositoryGraphBuilder(repo_root=project_root.parent / "DocuCharm")
        graph = builder.build(units)
        graph.save_json(graph_path)

    print(f"\n================================================================================")
    print(f"  CodeSage: Repository Graph Reranking Comparative Evaluation")
    print(f"  Evaluation Set: {len(queries)} queries (split: {args.split}, top_k: {args.top_k})")
    print(f"  Reranker Weights: alpha={args.alpha}, beta={args.beta}")
    print(f"================================================================================\n")

    # 1. Base TF-IDF & Graph Reranked TF-IDF
    print(">> [1/3] Evaluating TF-IDF Baseline vs TF-IDF + Graph Reranker...")
    tfidf_retriever = TfidfRetriever(include_metadata=True)
    tfidf_retriever.fit_from_jsonl(index_path)
    tfidf_results, tfidf_agg, tfidf_lat = benchmark_system(
        "TF-IDF (Base)", tfidf_retriever, queries, args.top_k, args.split
    )

    tfidf_reranker = GraphContextReranker(
        tfidf_retriever, graph, alpha=args.alpha, beta=args.beta, candidate_pool_k=25
    )
    tfidf_gr_results, tfidf_gr_agg, tfidf_gr_lat = benchmark_system(
        "TF-IDF + Graph Reranker", tfidf_reranker, queries, args.top_k, args.split
    )

    # 2. Base General Semantic & Graph Reranked Semantic
    print("\n>> [2/3] Evaluating Semantic (all-MiniLM-L6-v2) vs Semantic + Graph Reranker...")
    semantic_retriever = SemanticRetriever(cache_dir=cache_dir, include_metadata=True)
    semantic_retriever.fit_from_jsonl(index_path)
    sem_results, sem_agg, sem_lat = benchmark_system(
        "Semantic (Base)", semantic_retriever, queries, args.top_k, args.split
    )

    sem_reranker = GraphContextReranker(
        semantic_retriever, graph, alpha=args.alpha, beta=args.beta, candidate_pool_k=25
    )
    sem_gr_results, sem_gr_agg, sem_gr_lat = benchmark_system(
        "Semantic + Graph Reranker", sem_reranker, queries, args.top_k, args.split
    )

    # 3. Base Code Transformer & Graph Reranked Code Transformer
    print("\n>> [3/3] Evaluating Code Transformer (st-codesearch) vs Code + Graph Reranker...")
    code_retriever = CodeTransformerRetriever(cache_dir=cache_dir, include_metadata=True)
    code_retriever.fit_from_jsonl(index_path)
    code_results, code_agg, code_lat = benchmark_system(
        "Code Transformer (Base)", code_retriever, queries, args.top_k, args.split
    )

    code_reranker = GraphContextReranker(
        code_retriever, graph, alpha=args.alpha, beta=args.beta, candidate_pool_k=25
    )
    code_gr_results, code_gr_agg, code_gr_lat = benchmark_system(
        "Code Transformer + Graph Reranker", code_reranker, queries, args.top_k, args.split
    )

    # Compute comparative summaries
    models = [
        ("TF-IDF", tfidf_agg, tfidf_gr_agg, tfidf_lat, tfidf_gr_lat),
        ("Semantic (all-MiniLM)", sem_agg, sem_gr_agg, sem_lat, sem_gr_lat),
        ("Code Transformer (st-codesearch)", code_agg, code_gr_agg, code_lat, code_gr_lat),
    ]

    print("\n" + "=" * 96)
    print(f"{'RETRIEVAL SYSTEM':<34} | {'MRR':<7} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'nDCG@5':<7} | {'Latency':<8}")
    print("-" * 96)

    comparison_data: List[Dict[str, Any]] = []

    for name, base, reranked, b_lat, r_lat in models:
        mean_b_lat = sum(b_lat) / len(b_lat) if b_lat else 0.0
        mean_r_lat = sum(r_lat) / len(r_lat) if r_lat else 0.0
        overhead_ms = mean_r_lat - mean_b_lat
        overhead_pct = ((mean_r_lat - mean_b_lat) / mean_b_lat * 100.0) if mean_b_lat > 0 else 0.0

        d_mrr = reranked.mrr - base.mrr
        d_r1 = reranked.mean_recall_at_1 - base.mean_recall_at_1
        d_r5 = reranked.mean_recall_at_5 - base.mean_recall_at_5
        d_r10 = reranked.mean_recall_at_10 - base.mean_recall_at_10
        d_ndcg5 = reranked.mean_ndcg_at_5 - base.mean_ndcg_at_5

        print(f"{name + ' (Baseline)':<34} | {base.mrr:<7.4f} | {base.mean_recall_at_1:<7.4f} | {base.mean_recall_at_5:<7.4f} | {base.mean_recall_at_10:<7.4f} | {base.mean_ndcg_at_5:<7.4f} | {mean_b_lat:<6.2f}ms")
        print(f"{name + ' + Graph Rerank':<34} | {reranked.mrr:<7.4f} | {reranked.mean_recall_at_1:<7.4f} | {reranked.mean_recall_at_5:<7.4f} | {reranked.mean_recall_at_10:<7.4f} | {reranked.mean_ndcg_at_5:<7.4f} | {mean_r_lat:<6.2f}ms")
        print(f"{'  Delta (Net Improvement)':<34} | {format_delta(d_mrr):<7} | {format_delta(d_r1):<7} | {format_delta(d_r5):<7} | {format_delta(d_r10):<7} | {format_delta(d_ndcg5):<7} | {format_delta(overhead_ms)}ms")
        print("-" * 96)

        comparison_data.append({
            "model_family": name,
            "base_metrics": base.to_dict(),
            "reranked_metrics": reranked.to_dict(),
            "deltas": {
                "delta_mrr": round(d_mrr, 4),
                "delta_recall_1": round(d_r1, 4),
                "delta_recall_5": round(d_r5, 4),
                "delta_recall_10": round(d_r10, 4),
                "delta_ndcg_5": round(d_ndcg5, 4),
            },
            "latency": {
                "base_mean_ms": round(mean_b_lat, 3),
                "reranked_mean_ms": round(mean_r_lat, 3),
                "overhead_ms": round(overhead_ms, 3),
                "overhead_pct": round(overhead_pct, 1),
            },
        })

    # Export JSON
    if args.export_json:
        out_json = Path(args.export_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "evaluation_split": args.split,
            "top_k": args.top_k,
            "reranker_weights": {"alpha": args.alpha, "beta": args.beta},
            "comparisons": comparison_data,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nExported complete experiment report to: {out_json}")

    # Export CSV
    if args.export_csv:
        out_csv = Path(args.export_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model Family", "Configuration", "MRR", "Recall@1", "Recall@5",
                "Recall@10", "nDCG@5", "nDCG@10", "Mean Latency (ms)", "Latency Overhead (ms)"
            ])
            for item in comparison_data:
                name = item["model_family"]
                b = item["base_metrics"]
                r = item["reranked_metrics"]
                lat = item["latency"]
                writer.writerow([
                    name, "Base", b["mrr"], b["mean_recall_at_1"], b["mean_recall_at_5"],
                    b["mean_recall_at_10"], b["mean_ndcg_at_5"], b["mean_ndcg_at_10"], lat["base_mean_ms"], 0.0
                ])
                writer.writerow([
                    name, "Graph Reranked", r["mrr"], r["mean_recall_at_1"], r["mean_recall_at_5"],
                    r["mean_recall_at_10"], r["mean_ndcg_at_5"], r["mean_ndcg_at_10"], lat["reranked_mean_ms"], lat["overhead_ms"]
                ])
        print(f"Exported comparative CSV to: {out_csv}")


if __name__ == "__main__":
    main()
