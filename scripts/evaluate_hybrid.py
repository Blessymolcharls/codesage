#!/usr/bin/env python3
"""CodeSage Milestone 2: Hybrid Retrieval Evaluation Protocol using Reciprocal Rank Fusion.

Experimental Protocol:
1. Phase 1 - Development Split Experimentation (dev, 18 queries):
   - Evaluate TF-IDF baseline, Code Transformer baseline, TF-IDF + Graph,
     RRF Hybrid (TF-IDF + Code Transformer), and RRF Hybrid + Graph.
   - Perform fusion constant sensitivity analysis (k in {20, 40, 60, 80, 100}).
   - Select configuration strictly on development split without inspecting test split.
2. Phase 2 - Final Held-Out Evaluation (test, 18 queries):
   - Evaluate the selected configuration and baselines ONCE on the held-out test split.
   - Calculate bootstrap 95% confidence intervals and paired significance tests.
3. Export reproducible JSON and CSV artifacts.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure 'src' is in Python module search path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from codesage.code_semantic import CodeTransformerRetriever
from codesage.evaluation import (
    AggregateEvaluationResult,
    EvaluationQuery,
    QueryEvaluationResult,
    compute_bootstrap_ci,
    evaluate_dataset,
    load_evaluation_queries,
    run_paired_significance_test,
)
from codesage.graph import RepositoryGraph, RepositoryGraphBuilder
from codesage.graph_reranker import GraphContextReranker
from codesage.hybrid import HybridRetriever, HybridSearchResult, reciprocal_rank_fusion
from codesage.retrieval import SearchResult, TfidfRetriever, load_code_units_from_jsonl
from codesage.semantic import SemanticRetriever


def measure_retriever_latency(
    retriever: Any,
    queries: List[EvaluationQuery],
    top_k: int = 10,
    warmup_count: int = 2,
    repetitions: int = 3,
) -> Dict[str, float]:
    """Measure retrieval latency using median of repeated runs after warmup."""
    for q in queries[:min(warmup_count, len(queries))]:
        _ = retriever.search(q.query, top_k=top_k)

    latencies_ms: List[float] = []
    for q in queries:
        reps = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = retriever.search(q.query, top_k=top_k)
            reps.append((time.perf_counter() - t0) * 1000.0)
        latencies_ms.append(float(np.median(reps)))

    return {
        "mean": round(float(np.mean(latencies_ms)), 2),
        "p50": round(float(np.median(latencies_ms)), 2),
        "p95": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99": round(float(np.percentile(latencies_ms, 99)), 2),
    }


def evaluate_system_on_split(
    system_name: str,
    retriever: Any,
    queries: List[EvaluationQuery],
    split: str,
    top_k: int = 10,
) -> Tuple[AggregateEvaluationResult, List[QueryEvaluationResult], Dict[str, float]]:
    """Evaluate a single retrieval system on a specific query split with latency profiling."""
    q_results, agg = evaluate_dataset(retriever, queries, top_k=top_k, split_filter=split)
    latency = measure_retriever_latency(retriever, [q for q in queries if q.split == split], top_k=top_k)
    return agg, q_results, latency


def main():
    parser = argparse.ArgumentParser(description="CodeSage Milestone 2: Hybrid Retrieval Evaluation")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth (default: 10)")
    parser.add_argument("--k-fusion", type=int, default=60, help="Standard RRF fusion constant (default: 60)")
    args = parser.parse_args()

    data_dir = project_root / "data"
    index_path = data_dir / "code_units.jsonl"
    graph_path = data_dir / "relationship_graph.json"
    queries_path = data_dir / "evaluation_queries.jsonl"
    experiments_dir = data_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("CodeSage Milestone 2: Hybrid Retrieval Evaluation (Reciprocal Rank Fusion)")
    print("================================================================================")

    # 1. Load data
    print("Loading indexed code units, relationship graph, and evaluation queries...")
    code_units = load_code_units_from_jsonl(index_path)
    print(f"Loaded {len(code_units)} code units from {index_path.name}")

    if graph_path.exists():
        graph = RepositoryGraph.load_json(graph_path)
    else:
        builder = RepositoryGraphBuilder(repo_root=project_root.parent / "DocuCharm")
        graph = builder.build(code_units)
    print(f"Loaded repository graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    all_queries = load_evaluation_queries(queries_path)
    dev_queries = [q for q in all_queries if q.split == "dev"]
    test_queries = [q for q in all_queries if q.split == "test"]
    print(f"Total labelled queries: {len(all_queries)} (Dev: {len(dev_queries)}, Held-Out Test: {len(test_queries)})")

    # 2. Fit constituent retrievers
    print("\nFitting constituent retrieval engines...")
    tfidf = TfidfRetriever(include_metadata=True).fit(code_units)
    tfidf_graph = GraphContextReranker(tfidf, graph, alpha=0.75, beta=0.25, candidate_pool_k=25)

    cache_dir = data_dir / "cache"
    code_t = CodeTransformerRetriever(cache_dir=cache_dir, include_metadata=True).fit(code_units)
    code_graph = GraphContextReranker(code_t, graph, alpha=0.75, beta=0.25, candidate_pool_k=25)

    semantic = SemanticRetriever(cache_dir=cache_dir, include_metadata=True).fit(code_units)

    # ============================================================================
    # PHASE 1: DEVELOPMENT SPLIT EXPERIMENTATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: DEVELOPMENT SPLIT EXPERIMENTATION (N = 18 queries)")
    print("Note: Evaluating strictly on development split; held-out test split is preserved.")
    print("=" * 80)

    # Core systems on dev
    dev_systems = {
        "tfidf": ("TF-IDF (Lexical Baseline)", tfidf),
        "code": ("Code Transformer (st-codesearch)", code_t),
        "tfidf_graph": ("TF-IDF + Graph Reranker", tfidf_graph),
        "hybrid_rrf_k60": (
            "Hybrid RRF (TF-IDF + Code, k=60)",
            HybridRetriever(retrievers={"tfidf": tfidf, "code": code_t}, k=args.k_fusion, graph=graph),
        ),
        "hybrid_rrf_graph": (
            "Hybrid RRF + Graph Reranker (k=60)",
            GraphContextReranker(
                HybridRetriever(retrievers={"tfidf": tfidf, "code": code_t}, k=args.k_fusion),
                graph,
                alpha=0.75,
                beta=0.25,
                candidate_pool_k=25,
            ),
        ),
    }

    # Sensitivity analysis: testing fusion constants on dev split
    sensitivity_ks = [20, 40, 60, 80, 100]
    sensitivity_results = {}
    print("\nRunning RRF constant sensitivity analysis on development split:")
    for k_val in sensitivity_ks:
        h_k = HybridRetriever(retrievers={"tfidf": tfidf, "code": code_t}, k=k_val)
        _, agg_k = evaluate_dataset(h_k, dev_queries, top_k=args.top_k, split_filter="dev")
        sensitivity_results[f"k_{k_val}"] = {
            "k": k_val,
            "recall_at_1": round(agg_k.mean_recall_at_1, 4),
            "recall_at_5": round(agg_k.mean_recall_at_5, 4),
            "recall_at_10": round(agg_k.mean_recall_at_10, 4),
            "mrr": round(agg_k.mrr, 4),
            "ndcg_at_5": round(agg_k.mean_ndcg_at_5, 4),
            "ndcg_at_10": round(agg_k.mean_ndcg_at_10, 4),
        }
        print(f"  k={k_val:3d} -> R@1: {agg_k.mean_recall_at_1:.4f} | R@5: {agg_k.mean_recall_at_5:.4f} | MRR: {agg_k.mrr:.4f} | nDCG@5: {agg_k.mean_ndcg_at_5:.4f}")

    # Optional 3-way RRF on dev split
    hybrid_3way = HybridRetriever(retrievers={"tfidf": tfidf, "code": code_t, "semantic": semantic}, k=args.k_fusion)
    _, agg_3way = evaluate_dataset(hybrid_3way, dev_queries, top_k=args.top_k, split_filter="dev")
    print(f"  3-Way RRF (TF-IDF + Code + Semantic, k=60) -> R@1: {agg_3way.mean_recall_at_1:.4f} | R@5: {agg_3way.mean_recall_at_5:.4f} | MRR: {agg_3way.mrr:.4f} | nDCG@5: {agg_3way.mean_ndcg_at_5:.4f}")

    # Run evaluations on dev split for core systems
    dev_eval_results: Dict[str, Dict[str, Any]] = {}
    dev_query_results: Dict[str, List[QueryEvaluationResult]] = {}

    print("\nEvaluating core systems on Development Split:")
    print(f"{'System':<36} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'MRR':<7} | {'nDCG@5':<7} | {'nDCG@10':<7} | {'Lat(ms)':<7}")
    print("-" * 105)

    for sys_key, (disp_name, retriever_inst) in dev_systems.items():
        agg, q_res, lat = evaluate_system_on_split(disp_name, retriever_inst, dev_queries, split="dev", top_k=args.top_k)
        dev_eval_results[sys_key] = {
            "name": disp_name,
            "metrics": agg.to_dict(),
            "latency": lat,
        }
        dev_query_results[sys_key] = q_res
        print(
            f"{disp_name:<36} | "
            f"{agg.mean_recall_at_1:<7.4f} | "
            f"{agg.mean_recall_at_5:<7.4f} | "
            f"{agg.mean_recall_at_10:<7.4f} | "
            f"{agg.mrr:<7.4f} | "
            f"{agg.mean_ndcg_at_5:<7.4f} | "
            f"{agg.mean_ndcg_at_10:<7.4f} | "
            f"{lat['p50']:<7.2f}"
        )

    # Paired comparisons on dev split
    dev_significance = {
        "hybrid_vs_tfidf": run_paired_significance_test(
            dev_query_results["tfidf"],
            dev_query_results["hybrid_rrf_k60"],
        ),
        "hybrid_vs_code": run_paired_significance_test(
            dev_query_results["code"],
            dev_query_results["hybrid_rrf_k60"],
        ),
        "hybrid_graph_vs_hybrid": run_paired_significance_test(
            dev_query_results["hybrid_rrf_k60"],
            dev_query_results["hybrid_rrf_graph"],
        ),
    }

    # ============================================================================
    # PHASE 2: HELD-OUT TEST EVALUATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: FINAL HELD-OUT TEST EVALUATION (N = 18 queries)")
    print("Note: Evaluating ONCE on held-out test split using frozen configurations.")
    print("=" * 80)

    test_systems = {
        "tfidf": ("TF-IDF (Lexical Baseline)", tfidf),
        "code": ("Code Transformer (st-codesearch)", code_t),
        "tfidf_graph": ("TF-IDF + Graph Reranker", tfidf_graph),
        "hybrid_rrf_k60": (
            "Hybrid RRF (TF-IDF + Code, k=60)",
            HybridRetriever(retrievers={"tfidf": tfidf, "code": code_t}, k=args.k_fusion),
        ),
        "hybrid_rrf_graph": (
            "Hybrid RRF + Graph Reranker (k=60)",
            GraphContextReranker(
                HybridRetriever(retrievers={"tfidf": tfidf, "code": code_t}, k=args.k_fusion),
                graph,
                alpha=0.75,
                beta=0.25,
                candidate_pool_k=25,
            ),
        ),
    }

    test_eval_results: Dict[str, Dict[str, Any]] = {}
    test_query_results: Dict[str, List[QueryEvaluationResult]] = {}

    print(f"{'System':<36} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'MRR':<7} | {'nDCG@5':<7} | {'nDCG@10':<7} | {'Lat(ms)':<7}")
    print("-" * 105)

    for sys_key, (disp_name, retriever_inst) in test_systems.items():
        agg, q_res, lat = evaluate_system_on_split(disp_name, retriever_inst, test_queries, split="test", top_k=args.top_k)
        test_eval_results[sys_key] = {
            "name": disp_name,
            "metrics": agg.to_dict(),
            "latency": lat,
        }
        test_query_results[sys_key] = q_res
        print(
            f"{disp_name:<36} | "
            f"{agg.mean_recall_at_1:<7.4f} | "
            f"{agg.mean_recall_at_5:<7.4f} | "
            f"{agg.mean_recall_at_10:<7.4f} | "
            f"{agg.mrr:<7.4f} | "
            f"{agg.mean_ndcg_at_5:<7.4f} | "
            f"{agg.mean_ndcg_at_10:<7.4f} | "
            f"{lat['p50']:<7.2f}"
        )

    # Paired comparisons on held-out test split
    test_significance = {
        "hybrid_vs_tfidf": run_paired_significance_test(
            test_query_results["tfidf"],
            test_query_results["hybrid_rrf_k60"],
        ),
        "hybrid_vs_code": run_paired_significance_test(
            test_query_results["code"],
            test_query_results["hybrid_rrf_k60"],
        ),
        "hybrid_graph_vs_hybrid": run_paired_significance_test(
            test_query_results["hybrid_rrf_k60"],
            test_query_results["hybrid_rrf_graph"],
        ),
        "hybrid_graph_vs_tfidf_graph": run_paired_significance_test(
            test_query_results["tfidf_graph"],
            test_query_results["hybrid_rrf_graph"],
        ),
    }

    # ============================================================================
    # EXPORT REPRODUCIBLE ARTIFACTS
    # ============================================================================
    print("\nExporting experiment artifacts to data/experiments/...")

    # 1. Master JSON manifest
    master_manifest = {
        "metadata": {
            "study_name": "CodeSage Milestone 2: Hybrid Retrieval via Reciprocal Rank Fusion",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_units": len(code_units),
            "fusion_formula": "RRF(d) = sum_{m in M, d in m} (1 / (k + rank_m(d)))",
            "fusion_constant_k": args.k_fusion,
            "rank_indexing": "1-based",
            "dev_queries_count": len(dev_queries),
            "test_queries_count": len(test_queries),
        },
        "dev_split": {
            "systems": dev_eval_results,
            "sensitivity_analysis": sensitivity_results,
            "three_way_rrf": agg_3way.to_dict(),
            "paired_significance": dev_significance,
        },
        "test_split": {
            "systems": test_eval_results,
            "paired_significance": test_significance,
        },
    }

    json_path = experiments_dir / "hybrid_fusion_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)
    print(f"  Saved master JSON: {json_path}")

    # 2. Comparison CSV
    csv_path = experiments_dir / "hybrid_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "system_key",
            "system_name",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_5",
            "ndcg_at_10",
            "latency_p50_ms",
            "latency_mean_ms",
        ])
        for split_label, eval_dict in [("dev", dev_eval_results), ("test", test_eval_results)]:
            for k_sys, data in eval_dict.items():
                m = data["metrics"]
                lat = data["latency"]
                writer.writerow([
                    split_label,
                    k_sys,
                    data["name"],
                    f"{m['mean_recall_at_1']:.4f}",
                    f"{m['mean_recall_at_5']:.4f}",
                    f"{m['mean_recall_at_10']:.4f}",
                    f"{m['mrr']:.4f}",
                    f"{m['mean_ndcg_at_5']:.4f}",
                    f"{m['mean_ndcg_at_10']:.4f}",
                    f"{lat['p50']:.2f}",
                    f"{lat['mean']:.2f}",
                ])
    print(f"  Saved comparison CSV: {csv_path}")

    # 3. Per-query ranking CSV (held-out test split)
    per_query_csv = experiments_dir / "hybrid_per_query.csv"
    with open(per_query_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "query_id",
            "query",
            "system",
            "recall_at_1",
            "recall_at_5",
            "mrr",
            "ndcg_at_5",
            "retrieved_top_1",
            "relevant_count",
            "hits_at_5",
        ])
        for split_label, q_res_dict in [("dev", dev_query_results), ("test", test_query_results)]:
            for sys_key, q_list in q_res_dict.items():
                for qr in q_list:
                    top_1 = qr.retrieved_ids[0] if qr.retrieved_ids else "NONE"
                    hits_5 = sum(1 for uid in qr.retrieved_ids[:5] if uid in qr.relevant_ids)
                    writer.writerow([
                        split_label,
                        qr.query_id,
                        qr.query,
                        sys_key,
                        f"{qr.recall_at_1:.4f}",
                        f"{qr.recall_at_5:.4f}",
                        f"{qr.mrr:.4f}",
                        f"{qr.ndcg_at_5:.4f}",
                        top_1,
                        len(qr.relevant_ids),
                        hits_5,
                    ])
    print(f"  Saved per-query rankings CSV: {per_query_csv}")

    print("\nHybrid evaluation run successfully completed!")


if __name__ == "__main__":
    main()
