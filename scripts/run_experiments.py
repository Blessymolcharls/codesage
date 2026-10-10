#!/usr/bin/env python3
"""CodeSage Milestone 5: Research Evaluation and Controlled Ablation Runner.

Executes a unified, reproducible evaluation protocol across:
1. TF-IDF Lexical Retrieval (standard, char n-grams, no-metadata)
2. General-Purpose Sentence Embeddings (all-MiniLM-L6-v2: standard vs no-metadata)
3. Code-Specific Pretrained Transformer (st-codesearch-distilroberta-base: standard vs no-metadata)

Evaluates on:
- Development set (dev, 18 queries)
- Held-out test set (test, 18 queries)
- Full dataset (all, 40 queries including 4 negative controls)
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
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
from codesage.retrieval import TfidfRetriever, load_code_units_from_jsonl
from codesage.semantic import SemanticRetriever


def get_cache_disk_footprint(cache_dir: Path) -> Dict[str, float]:
    """Calculate disk footprints for cached embeddings in KB."""
    sizes: Dict[str, float] = {}
    if cache_dir.exists():
        for p in cache_dir.glob("*.npz"):
            sizes[p.name] = round(p.stat().st_size / 1024.0, 2)
    return sizes


def run_single_experiment(
    exp_name: str,
    retriever_factory: Any,
    index_path: Path,
    queries: List[EvaluationQuery],
    top_k: int = 10,
    splits: Tuple[str, ...] = ("dev", "test", "all"),
) -> Dict[str, Any]:
    """Index once, measure latency, and evaluate across specified splits."""
    print(f"--> Running Experiment: [{exp_name}] ...")
    t0 = time.perf_counter()
    retriever = retriever_factory()
    retriever.fit_from_jsonl(index_path)
    indexing_time_s = time.perf_counter() - t0

    # Measure latency across all queries
    latencies_ms: List[float] = []
    for q in queries:
        q_start = time.perf_counter()
        _ = retriever.search(q.query, top_k=top_k)
        latencies_ms.append((time.perf_counter() - q_start) * 1000.0)

    split_results: Dict[str, Any] = {}
    query_results_map: Dict[str, List[QueryEvaluationResult]] = {}

    for s in splits:
        q_results, agg = evaluate_dataset(retriever, queries, top_k=top_k, split_filter=s)
        split_results[s] = agg.to_dict()
        query_results_map[s] = q_results

    return {
        "experiment_name": exp_name,
        "indexing_time_seconds": round(indexing_time_s, 4),
        "latency_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2),
            "p50": round(float(np.median(latencies_ms)), 2),
            "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            "p99": round(float(np.percentile(latencies_ms, 99)), 2),
        },
        "splits": split_results,
        "_query_results": query_results_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeSage Milestone 5: Research Evaluation & Ablation Runner")
    parser.add_argument(
        "--queries",
        type=str,
        default=str(project_root / "data" / "evaluation_queries.jsonl"),
        help="Path to evaluation queries JSONL",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=str(project_root / "data" / "code_units.jsonl"),
        help="Path to code units JSONL index",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data" / "experiments"),
        help="Output directory for study results and tables",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Evaluation top-k retrieval depth (default: 10)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of all dense embeddings bypassing cache",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    index_path = Path(args.index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = project_root / "data" / "cache"

    print("=" * 80)
    print("CodeSage Milestone 5: Research Evaluation & Ablation Study Runner")
    print("=" * 80)

    # 1. Audit experimental setup
    queries = load_evaluation_queries(queries_path)
    indexed_units = load_code_units_from_jsonl(index_path)
    indexed_ids = {u["unit_id"] for u in indexed_units}

    dev_queries = [q for q in queries if q.split == "dev"]
    test_queries = [q for q in queries if q.split == "test"]
    neg_queries = [q for q in queries if q.split == "negative_control"]

    # Verify zero overlap between dev and test queries
    dev_texts = {q.query.lower().strip() for q in dev_queries}
    test_texts = {q.query.lower().strip() for q in test_queries}
    overlap = dev_texts.intersection(test_texts)
    assert not overlap, f"Fatal: Overlap found between dev and test queries: {overlap}"

    # Verify all relevant unit IDs are grounded in actual source code units
    missing_groundings: Dict[str, List[str]] = {}
    for q in queries:
        for rid in q.relevant_unit_ids:
            if rid not in indexed_ids:
                missing_groundings.setdefault(q.query_id, []).append(rid)

    assert not missing_groundings, f"Fatal: Unindexed relevance annotations: {missing_groundings}"

    print(f"Experimental Audit Passed:")
    print(f"  * Code Units Indexed : {len(indexed_ids)} (DocuCharm Dart frontend + JS backend)")
    print(f"  * Total Queries      : {len(queries)}")
    print(f"  * Dev Set (Tuning)   : {len(dev_queries)} queries (Zero overlap with test)")
    print(f"  * Test Set (Held-Out): {len(test_queries)} queries (Strictly isolated)")
    print(f"  * Negative Controls  : {len(neg_queries)} queries (Zero relevant units)")
    print(f"  * Top-K Depth        : {args.top_k}")
    print("-" * 80)

    # 2. Define controlled experiment matrix
    experiments_def = [
        # Primary models
        ("tfidf_standard", lambda: TfidfRetriever(use_char_ngrams=False, include_metadata=True)),
        ("semantic_standard", lambda: SemanticRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=True)),
        ("code_transformer_standard", lambda: CodeTransformerRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=True)),
        # Ablations
        ("tfidf_char_ngrams", lambda: TfidfRetriever(use_char_ngrams=True, include_metadata=True)),
        ("tfidf_no_metadata", lambda: TfidfRetriever(use_char_ngrams=False, include_metadata=False)),
        ("semantic_no_metadata", lambda: SemanticRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=False)),
        ("code_transformer_no_metadata", lambda: CodeTransformerRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=False)),
    ]

    all_results: Dict[str, Any] = {}
    raw_query_results_all: Dict[str, Dict[str, List[QueryEvaluationResult]]] = {}

    for exp_name, factory in experiments_def:
        res = run_single_experiment(
            exp_name=exp_name,
            retriever_factory=factory,
            index_path=index_path,
            queries=queries,
            top_k=args.top_k,
            splits=("dev", "test", "all"),
        )
        raw_query_results_all[exp_name] = res.pop("_query_results")
        all_results[exp_name] = res

    # 3. Significance testing on held-out test split and full split
    significance_reports: Dict[str, Any] = {}
    for split_key in ("test", "all"):
        significance_reports[split_key] = {
            "code_vs_tfidf": run_paired_significance_test(
                raw_query_results_all["tfidf_standard"][split_key],
                raw_query_results_all["code_transformer_standard"][split_key],
            ),
            "code_vs_semantic": run_paired_significance_test(
                raw_query_results_all["semantic_standard"][split_key],
                raw_query_results_all["code_transformer_standard"][split_key],
            ),
            "semantic_vs_tfidf": run_paired_significance_test(
                raw_query_results_all["tfidf_standard"][split_key],
                raw_query_results_all["semantic_standard"][split_key],
            ),
            "tfidf_char_vs_word": run_paired_significance_test(
                raw_query_results_all["tfidf_standard"][split_key],
                raw_query_results_all["tfidf_char_ngrams"][split_key],
            ),
            "code_meta_vs_nometa": run_paired_significance_test(
                raw_query_results_all["code_transformer_no_metadata"][split_key],
                raw_query_results_all["code_transformer_standard"][split_key],
            ),
            "semantic_meta_vs_nometa": run_paired_significance_test(
                raw_query_results_all["semantic_no_metadata"][split_key],
                raw_query_results_all["semantic_standard"][split_key],
            ),
        }

    # 4. Storage and Model Footprints
    storage_info = {
        "models": {
            "tfidf": {"parameters": "0", "embedding_dim": "Sparse vocab", "weight_mb": 0.0},
            "all-MiniLM-L6-v2": {"parameters": "22.7M", "embedding_dim": 384, "weight_mb": 86.7},
            "st-codesearch-distilroberta": {"parameters": "82.1M", "embedding_dim": 768, "weight_mb": 313.3},
        },
        "embedding_caches_kb": get_cache_disk_footprint(cache_dir),
    }

    study_manifest = {
        "study_metadata": {
            "dataset_version": "1.2.0-milestone5",
            "evaluated_repo": "DocuCharm",
            "indexed_units_count": len(indexed_ids),
            "queries_count": len(queries),
            "dev_queries_count": len(dev_queries),
            "test_queries_count": len(test_queries),
            "negative_control_count": len(neg_queries),
            "top_k": args.top_k,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "experiments": all_results,
        "paired_significance": significance_reports,
        "storage": storage_info,
    }

    # Export study JSON
    json_path = output_dir / "study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(study_manifest, f, indent=2)
    print(f"\n[OK] Exported full research study manifest: {json_path}")

    # Export Ablation CSV
    csv_path = output_dir / "ablation_study.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "split",
            "evaluated_queries",
            "recall@1",
            "recall@1_95ci",
            "recall@5",
            "recall@5_95ci",
            "recall@10",
            "recall@10_95ci",
            "mrr",
            "mrr_95ci",
            "ndcg@5",
            "ndcg@5_95ci",
            "ndcg@10",
            "ndcg@10_95ci",
            "indexing_s",
            "latency_p50_ms",
        ])
        for exp_name, exp_data in all_results.items():
            idx_s = exp_data["indexing_time_seconds"]
            lat_p50 = exp_data["latency_ms"]["p50"]
            for split_name, s_data in exp_data["splits"].items():
                ci = s_data.get("confidence_intervals_95", {})
                r1_ci = f"[{ci.get('recall_at_1', (0,0))[0]:.3f}, {ci.get('recall_at_1', (0,0))[1]:.3f}]" if "recall_at_1" in ci else ""
                r5_ci = f"[{ci.get('recall_at_5', (0,0))[0]:.3f}, {ci.get('recall_at_5', (0,0))[1]:.3f}]" if "recall_at_5" in ci else ""
                r10_ci = f"[{ci.get('recall_at_10', (0,0))[0]:.3f}, {ci.get('recall_at_10', (0,0))[1]:.3f}]" if "recall_at_10" in ci else ""
                mrr_ci = f"[{ci.get('mrr', (0,0))[0]:.3f}, {ci.get('mrr', (0,0))[1]:.3f}]" if "mrr" in ci else ""
                n5_ci = f"[{ci.get('ndcg_at_5', (0,0))[0]:.3f}, {ci.get('ndcg_at_5', (0,0))[1]:.3f}]" if "ndcg_at_5" in ci else ""
                n10_ci = f"[{ci.get('ndcg_at_10', (0,0))[0]:.3f}, {ci.get('ndcg_at_10', (0,0))[1]:.3f}]" if "ndcg_at_10" in ci else ""

                writer.writerow([
                    exp_name,
                    split_name,
                    s_data["evaluated_queries"],
                    f"{s_data['mean_recall_at_1']:.4f}",
                    r1_ci,
                    f"{s_data['mean_recall_at_5']:.4f}",
                    r5_ci,
                    f"{s_data['mean_recall_at_10']:.4f}",
                    r10_ci,
                    f"{s_data['mrr']:.4f}",
                    mrr_ci,
                    f"{s_data['mean_ndcg_at_5']:.4f}",
                    n5_ci,
                    f"{s_data['mean_ndcg_at_10']:.4f}",
                    n10_ci,
                    idx_s,
                    lat_p50,
                ])
    print(f"[OK] Exported ablation comparison table: {csv_path}")

    # Export Per-Query Rankings CSV
    query_rankings_path = output_dir / "per_query_rankings.csv"
    with open(query_rankings_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_id",
            "split",
            "task_category",
            "query",
            "experiment",
            "recall@1",
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@5",
            "retrieved_top3",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp_name, split_dict in raw_query_results_all.items():
            for q_res in split_dict.get("all", []):
                writer.writerow({
                    "query_id": q_res.query_id,
                    "split": getattr(q_res, "split", "test"),
                    "task_category": q_res.task_category,
                    "query": q_res.query,
                    "experiment": exp_name,
                    "recall@1": f"{q_res.recall_at_1:.4f}",
                    "recall@5": f"{q_res.recall_at_5:.4f}",
                    "recall@10": f"{q_res.recall_at_10:.4f}",
                    "mrr": f"{q_res.mrr:.4f}",
                    "ndcg@5": f"{q_res.ndcg_at_5:.4f}",
                    "retrieved_top3": ";".join(q_res.retrieved_ids[:3]),
                })
    print(f"[OK] Exported per-query rankings CSV: {query_rankings_path}")

    # Print Formatted Results Table for Held-Out Test Set
    print("\n" + "=" * 90)
    print("HELD-OUT TEST SET RESULTS (18 Queries, Unseen during prompt / parameter tuning)")
    print("=" * 90)
    print(f"{'Experiment':<30} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'MRR':<7} | {'nDCG@5':<7} | {'Lat p50':<8}")
    print("-" * 90)
    for exp_name, exp_data in all_results.items():
        s = exp_data["splits"]["test"]
        p50 = exp_data["latency_ms"]["p50"]
        print(
            f"{exp_name:<30} | {s['mean_recall_at_1']:.4f} | {s['mean_recall_at_5']:.4f} | "
            f"{s['mean_recall_at_10']:.4f} | {s['mrr']:.4f} | {s['mean_ndcg_at_5']:.4f} | {p50:6.2f} ms"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
