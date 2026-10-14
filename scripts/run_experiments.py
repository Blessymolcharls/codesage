#!/usr/bin/env python3
"""CodeSage: Rigorous, Reproducible Retrieval Evaluation Study.

Executes a controlled evaluation protocol comparing:
1. Lexical Baseline: TF-IDF (word n-grams, char n-grams, no-metadata)
2. General-Purpose Dense Semantic: all-MiniLM-L6-v2 (standard vs no-metadata)
3. Code-Specific Transformer: st-codesearch-distilroberta-base (standard vs no-metadata)
4. Graph Context Reranking: applied to TF-IDF, Semantic, and Code Transformer

Evaluates strictly isolated splits:
- Development split (dev, 18 queries)
- Held-out test split (test, 18 queries)
- Negative control queries (4 queries evaluated separately for false alarm rates)
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
    NegativeControlReport,
    QueryEvaluationResult,
    compute_bootstrap_ci,
    evaluate_dataset,
    evaluate_negative_controls,
    load_evaluation_queries,
    run_paired_significance_test,
)
from codesage.graph import RepositoryGraph, RepositoryGraphBuilder
from codesage.graph_reranker import GraphContextReranker
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
    """Index once, measure latency with repeated trials, evaluate metrics and negative controls."""
    print(f"--> Running Experiment: [{exp_name}] ...")
    t0 = time.perf_counter()
    retriever = retriever_factory()
    if hasattr(retriever, "fit_from_jsonl") and not getattr(retriever, "units", None):
        retriever.fit_from_jsonl(index_path)
    indexing_time_s = time.perf_counter() - t0

    # Timing methodology: 1 warmup query + 3 timed repetitions per query taking median
    for q in queries[:min(3, len(queries))]:
        _ = retriever.search(q.query, top_k=top_k)

    latencies_ms: List[float] = []
    for q in queries:
        reps = []
        for _ in range(3):
            q_start = time.perf_counter()
            _ = retriever.search(q.query, top_k=top_k)
            reps.append((time.perf_counter() - q_start) * 1000.0)
        latencies_ms.append(float(np.median(reps)))

    split_results: Dict[str, Any] = {}
    query_results_map: Dict[str, List[QueryEvaluationResult]] = {}

    for s in splits:
        q_results, agg = evaluate_dataset(retriever, queries, top_k=top_k, split_filter=s)
        split_results[s] = agg.to_dict()
        query_results_map[s] = q_results

    # Negative control false-alarm analysis
    neg_report = evaluate_negative_controls(retriever, queries, top_k=5)

    return {
        "experiment_name": exp_name,
        "indexing_time_seconds": round(indexing_time_s, 4),
        "latency_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2),
            "p50": round(float(np.median(latencies_ms)), 2),
            "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            "p99": round(float(np.percentile(latencies_ms, 99)), 2),
            "measurement_method": "Median of 3 repetitions per query after warmup",
        },
        "splits": split_results,
        "negative_controls": neg_report.to_dict(),
        "_query_results": query_results_map,
    }


def analyze_failure_cases(
    queries: List[EvaluationQuery],
    results_map: Dict[str, Dict[str, List[QueryEvaluationResult]]],
) -> Dict[str, Any]:
    """Diagnose retrieval failure modes across query intents, models, and graph reranking."""
    q_by_id = {q.query_id: q for q in queries}

    tfidf_all = {r.query_id: r for r in results_map.get("tfidf_standard", {}).get("all", [])}
    sem_all = {r.query_id: r for r in results_map.get("semantic_standard", {}).get("all", [])}
    code_all = {r.query_id: r for r in results_map.get("code_transformer_standard", {}).get("all", [])}
    tfidf_gr = {r.query_id: r for r in results_map.get("tfidf_graph", {}).get("all", [])}
    sem_gr = {r.query_id: r for r in results_map.get("semantic_graph", {}).get("all", [])}
    code_gr = {r.query_id: r for r in results_map.get("code_transformer_graph", {}).get("all", [])}

    lexical_gaps: List[Dict[str, Any]] = []
    semantic_gaps: List[Dict[str, Any]] = []
    hard_queries: List[Dict[str, Any]] = []
    graph_improvements: List[Dict[str, Any]] = []
    graph_degradations: List[Dict[str, Any]] = []

    for qid, q in q_by_id.items():
        if q.split == "negative_control":
            continue

        t_r = tfidf_all.get(qid)
        s_r = sem_all.get(qid)
        c_r = code_all.get(qid)
        t_gr_r = tfidf_gr.get(qid)
        s_gr_r = sem_gr.get(qid)
        c_gr_r = code_gr.get(qid)

        if not t_r or not s_r or not c_r:
            continue

        # 1. Lexical gap: TF-IDF misses, neural succeeds
        if t_r.recall_at_5 == 0.0 and (s_r.recall_at_5 >= 0.5 or c_r.recall_at_5 >= 0.5):
            lexical_gaps.append({
                "query_id": qid,
                "split": q.split,
                "query": q.query,
                "relevant_units": q.relevant_unit_ids,
                "tfidf_r5": t_r.recall_at_5,
                "semantic_r5": s_r.recall_at_5,
                "code_r5": c_r.recall_at_5,
                "diagnosis": "Lexical vocabulary mismatch: query uses intent words absent from symbol names or code body.",
            })

        # 2. Semantic gap: TF-IDF succeeds, neural misses
        if t_r.recall_at_5 >= 0.5 and s_r.recall_at_5 == 0.0:
            semantic_gaps.append({
                "query_id": qid,
                "split": q.split,
                "query": q.query,
                "relevant_units": q.relevant_unit_ids,
                "tfidf_r5": t_r.recall_at_5,
                "semantic_r5": s_r.recall_at_5,
                "code_r5": c_r.recall_at_5,
                "diagnosis": "General semantic dilution: sentence embedding fails on code-specific identifiers.",
            })

        # 3. Hard queries: all 3 baseline engines fail to achieve Recall@5 >= 0.5
        if t_r.recall_at_5 == 0.0 and s_r.recall_at_5 == 0.0 and c_r.recall_at_5 == 0.0:
            hard_queries.append({
                "query_id": qid,
                "split": q.split,
                "query": q.query,
                "relevant_units": q.relevant_unit_ids,
                "diagnosis": "Extreme vocabulary and abstraction gap across all retrieval representations.",
            })

        # 4. Graph reranking improvement
        if t_gr_r and (t_gr_r.ndcg_at_5 > t_r.ndcg_at_5 + 0.05):
            graph_improvements.append({
                "query_id": qid,
                "model": "TF-IDF + Graph",
                "query": q.query,
                "base_ndcg5": t_r.ndcg_at_5,
                "reranked_ndcg5": t_gr_r.ndcg_at_5,
                "delta": round(t_gr_r.ndcg_at_5 - t_r.ndcg_at_5, 4),
                "diagnosis": "Syntax affinity boosted relevant co-defined or called units into top ranks.",
            })
        elif s_gr_r and (s_gr_r.ndcg_at_5 > s_r.ndcg_at_5 + 0.05):
            graph_improvements.append({
                "query_id": qid,
                "model": "Semantic + Graph",
                "query": q.query,
                "base_ndcg5": s_r.ndcg_at_5,
                "reranked_ndcg5": s_gr_r.ndcg_at_5,
                "delta": round(s_gr_r.ndcg_at_5 - s_r.ndcg_at_5, 4),
                "diagnosis": "Class-method membership affinity promoted target implementation unit.",
            })

        # 5. Graph reranking degradation
        if t_gr_r and (t_gr_r.ndcg_at_5 < t_r.ndcg_at_5 - 0.05):
            graph_degradations.append({
                "query_id": qid,
                "model": "TF-IDF + Graph",
                "query": q.query,
                "base_ndcg5": t_r.ndcg_at_5,
                "reranked_ndcg5": t_gr_r.ndcg_at_5,
                "delta": round(t_gr_r.ndcg_at_5 - t_r.ndcg_at_5, 4),
                "diagnosis": "Dense cluster of connected but non-relevant units crowded out relevant candidate.",
            })

    return {
        "summary": {
            "total_queries_evaluated": len(queries) - len([q for q in queries if q.split == "negative_control"]),
            "lexical_gap_queries_count": len(lexical_gaps),
            "semantic_gap_queries_count": len(semantic_gaps),
            "hard_queries_count": len(hard_queries),
            "graph_improvements_count": len(graph_improvements),
            "graph_degradations_count": len(graph_degradations),
        },
        "lexical_gap_queries": lexical_gaps,
        "semantic_gap_queries": semantic_gaps,
        "hard_queries": hard_queries,
        "graph_improvements": graph_improvements,
        "graph_degradations": graph_degradations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeSage: Rigorous Retrieval Evaluation Study")
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
        "--graph",
        type=str,
        default=str(project_root / "data" / "relationship_graph.json"),
        help="Path to relationship graph JSON",
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
    graph_path = Path(args.graph)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = project_root / "data" / "cache"

    print("=" * 90)
    print("CodeSage: Rigorous, Reproducible Retrieval Evaluation Study")
    print("=" * 90)

    # 1. Audit experimental dataset
    queries = load_evaluation_queries(queries_path)
    indexed_units = load_code_units_from_jsonl(index_path)
    indexed_ids = {u["unit_id"] for u in indexed_units}

    dev_queries = [q for q in queries if q.split == "dev"]
    test_queries = [q for q in queries if q.split == "test"]
    neg_queries = [q for q in queries if q.split == "negative_control"]

    dev_texts = {q.query.lower().strip() for q in dev_queries}
    test_texts = {q.query.lower().strip() for q in test_queries}
    overlap = dev_texts.intersection(test_texts)
    assert not overlap, f"Fatal: Dev and Test query overlap: {overlap}"

    missing_groundings: Dict[str, List[str]] = {}
    for q in queries:
        for rid in q.relevant_unit_ids:
            if rid not in indexed_ids:
                missing_groundings.setdefault(q.query_id, []).append(rid)
    assert not missing_groundings, f"Fatal: Unindexed relevance annotations: {missing_groundings}"

    # Load relationship graph
    if graph_path.exists():
        graph = RepositoryGraph.load_json(graph_path)
    else:
        builder = RepositoryGraphBuilder(repo_root=project_root.parent / "DocuCharm")
        graph = builder.build(indexed_units)
        graph.save_json(graph_path)

    print(f"Experimental Setup Audited:")
    print(f"  * Indexed Units     : {len(indexed_ids)} units (16 files: 80 Dart, 9 JavaScript)")
    print(f"  * Repository Graph  : {len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(graph.unresolved_references)} unresolved refs")
    print(f"  * Total Queries     : {len(queries)}")
    print(f"  * Dev Set (Tuning)  : {len(dev_queries)} queries (Zero overlap with test)")
    print(f"  * Test Set (Held-Out: {len(test_queries)} queries (Strictly isolated)")
    print(f"  * Negative Controls : {len(neg_queries)} queries (Explicitly evaluated)")
    print(f"  * Top-K Depth       : {args.top_k}")
    print("-" * 90)

    # Pre-fit base retrievers
    print("Fitting baseline retrievers...")
    tfidf_base = TfidfRetriever(use_char_ngrams=False, include_metadata=True).fit(indexed_units)
    sem_base = SemanticRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=True).fit(indexed_units)
    code_base = CodeTransformerRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=True).fit(indexed_units)

    # 2. Experiment Matrix
    experiments_def = [
        # Baselines
        ("tfidf_standard", lambda: tfidf_base),
        ("semantic_standard", lambda: sem_base),
        ("code_transformer_standard", lambda: code_base),
        # Graph Reranked Counterparts
        ("tfidf_graph", lambda: GraphContextReranker(tfidf_base, graph, alpha=0.75, beta=0.25, candidate_pool_k=25)),
        ("semantic_graph", lambda: GraphContextReranker(sem_base, graph, alpha=0.75, beta=0.25, candidate_pool_k=25)),
        ("code_transformer_graph", lambda: GraphContextReranker(code_base, graph, alpha=0.75, beta=0.25, candidate_pool_k=25)),
        # Controlled Ablations
        ("tfidf_char_ngrams", lambda: TfidfRetriever(use_char_ngrams=True, include_metadata=True).fit(indexed_units)),
        ("tfidf_no_metadata", lambda: TfidfRetriever(use_char_ngrams=False, include_metadata=False).fit(indexed_units)),
        ("semantic_no_metadata", lambda: SemanticRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=False).fit(indexed_units)),
        ("code_transformer_no_metadata", lambda: CodeTransformerRetriever(cache_dir=cache_dir, force_recompute=args.force_recompute, include_metadata=False).fit(indexed_units)),
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

    # 3. Paired Statistical Significance Testing
    significance_reports: Dict[str, Any] = {}
    for split_key in ("dev", "test", "all"):
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
            "tfidf_graph_vs_base": run_paired_significance_test(
                raw_query_results_all["tfidf_standard"][split_key],
                raw_query_results_all["tfidf_graph"][split_key],
            ),
            "semantic_graph_vs_base": run_paired_significance_test(
                raw_query_results_all["semantic_standard"][split_key],
                raw_query_results_all["semantic_graph"][split_key],
            ),
            "code_graph_vs_base": run_paired_significance_test(
                raw_query_results_all["code_transformer_standard"][split_key],
                raw_query_results_all["code_transformer_graph"][split_key],
            ),
        }

    # 4. Storage Footprints
    storage_info = {
        "models": {
            "tfidf": {"parameters": "0", "embedding_dim": "Sparse vocab", "weight_mb": 0.0},
            "all-MiniLM-L6-v2": {"parameters": "22.7M", "embedding_dim": 384, "weight_mb": 86.7},
            "st-codesearch-distilroberta": {"parameters": "82.1M", "embedding_dim": 768, "weight_mb": 313.3},
        },
        "embedding_caches_kb": get_cache_disk_footprint(cache_dir),
    }

    # 5. Failure Case Diagnosis
    failure_cases = analyze_failure_cases(queries, raw_query_results_all)

    # 6. Negative Controls Analysis Extraction
    negative_controls_summary = {
        exp_name: exp_data["negative_controls"]
        for exp_name, exp_data in all_results.items()
    }

    # Save Study Manifest
    study_manifest = {
        "study_metadata": {
            "dataset_version": "2.0.0-rigorous-study",
            "evaluated_repo": "DocuCharm",
            "indexed_units_count": len(indexed_ids),
            "queries_count": len(queries),
            "dev_queries_count": len(dev_queries),
            "test_queries_count": len(test_queries),
            "negative_control_count": len(neg_queries),
            "top_k": args.top_k,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "timing_methodology": "1 warmup iteration + 3 timed repetitions per query taking median latency",
        },
        "experiments": all_results,
        "paired_significance": significance_reports,
        "negative_controls_summary": negative_controls_summary,
        "failure_cases": failure_cases,
        "storage": storage_info,
    }

    json_path = output_dir / "study_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(study_manifest, f, indent=2)
    print(f"\n[OK] Exported full research study manifest: {json_path}")

    # Export Negative Controls CSV
    neg_csv_path = output_dir / "negative_controls_analysis.csv"
    with open(neg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "total_negatives",
            "mean_top1_score",
            "p50_top1_score",
            "max_top1_score",
            "rate_ge_0.2",
            "rate_ge_0.3",
            "rate_ge_0.4",
            "rate_ge_0.5",
            "rate_ge_0.6",
            "rate_ge_0.7",
        ])
        for exp_name, exp_data in all_results.items():
            nc = exp_data["negative_controls"]
            rates = nc["threshold_exceedance_rates"]
            writer.writerow([
                exp_name,
                nc["total_negative_queries"],
                nc["mean_top1_score"],
                nc["p50_top1_score"],
                nc["max_top1_score"],
                rates.get(">=0.2", 0.0),
                rates.get(">=0.3", 0.0),
                rates.get(">=0.4", 0.0),
                rates.get(">=0.5", 0.0),
                rates.get(">=0.6", 0.0),
                rates.get(">=0.7", 0.0),
            ])
    print(f"[OK] Exported negative controls analysis CSV: {neg_csv_path}")

    # Export Failure Cases JSON
    failure_path = output_dir / "failure_cases_analysis.json"
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_cases, f, indent=2)
    print(f"[OK] Exported failure cases analysis: {failure_path}")

    # Export Graph Reranking Results JSON & CSV
    graph_results_export = {
        "evaluation_split": "test",
        "top_k": args.top_k,
        "reranker_weights": {"alpha": 0.75, "beta": 0.25},
        "comparisons": [
            {
                "model_family": "TF-IDF",
                "base_metrics": all_results["tfidf_standard"]["splits"]["test"],
                "reranked_metrics": all_results["tfidf_graph"]["splits"]["test"],
            },
            {
                "model_family": "Semantic (all-MiniLM)",
                "base_metrics": all_results["semantic_standard"]["splits"]["test"],
                "reranked_metrics": all_results["semantic_graph"]["splits"]["test"],
            },
            {
                "model_family": "Code Transformer (st-codesearch)",
                "base_metrics": all_results["code_transformer_standard"]["splits"]["test"],
                "reranked_metrics": all_results["code_transformer_graph"]["splits"]["test"],
            },
        ],
    }
    graph_json_path = output_dir / "graph_reranking_results.json"
    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(graph_results_export, f, indent=2)
    print(f"[OK] Exported graph reranking JSON: {graph_json_path}")

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
            "latency_p50_ms",
        ])
        for exp_name, exp_data in all_results.items():
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

    # 7. Print Terminal Summary Tables
    print("\n" + "=" * 94)
    print("DEVELOPMENT SET RESULTS (18 Queries, Parameter & Prompt Tuning)")
    print("=" * 94)
    print(f"{'Experiment':<30} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'MRR':<7} | {'nDCG@5':<7} | {'Lat p50':<8}")
    print("-" * 94)
    for exp_name, exp_data in all_results.items():
        s = exp_data["splits"]["dev"]
        p50 = exp_data["latency_ms"]["p50"]
        print(
            f"{exp_name:<30} | {s['mean_recall_at_1']:.4f} | {s['mean_recall_at_5']:.4f} | "
            f"{s['mean_recall_at_10']:.4f} | {s['mrr']:.4f} | {s['mean_ndcg_at_5']:.4f} | {p50:6.2f} ms"
        )
    print("=" * 94)

    print("\n" + "=" * 94)
    print("HELD-OUT TEST SET RESULTS (18 Queries, Strictly Preserved for Final Evaluation)")
    print("=" * 94)
    print(f"{'Experiment':<30} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'MRR':<7} | {'nDCG@5':<7} | {'Lat p50':<8}")
    print("-" * 94)
    for exp_name, exp_data in all_results.items():
        s = exp_data["splits"]["test"]
        p50 = exp_data["latency_ms"]["p50"]
        print(
            f"{exp_name:<30} | {s['mean_recall_at_1']:.4f} | {s['mean_recall_at_5']:.4f} | "
            f"{s['mean_recall_at_10']:.4f} | {s['mrr']:.4f} | {s['mean_ndcg_at_5']:.4f} | {p50:6.2f} ms"
        )
    print("=" * 94)

    print("\n" + "=" * 94)
    print("NEGATIVE CONTROL FALSE ALARM ANALYSIS (4 Unrelated / Irrelevant Queries)")
    print("=" * 94)
    print(f"{'Experiment':<30} | {'Mean Top1':<9} | {'Max Top1':<9} | {'% >=0.3':<8} | {'% >=0.5':<8} | {'% >=0.7':<8}")
    print("-" * 94)
    for exp_name, exp_data in all_results.items():
        nc = exp_data["negative_controls"]
        r = nc["threshold_exceedance_rates"]
        print(
            f"{exp_name:<30} | {nc['mean_top1_score']:<9.4f} | {nc['max_top1_score']:<9.4f} | "
            f"{r.get('>=0.3', 0.0)*100:6.1f}% | {r.get('>=0.5', 0.0)*100:6.1f}% | {r.get('>=0.7', 0.0)*100:6.1f}%"
        )
    print("=" * 94)


if __name__ == "__main__":
    main()
