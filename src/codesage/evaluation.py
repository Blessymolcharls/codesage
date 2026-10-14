"""Evaluation harness and IR metrics for CodeSage code search."""

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class EvaluationQuery:
    """A labelled query with verified relevant code-unit IDs."""
    query_id: str
    query: str
    relevant_unit_ids: List[str]
    task_category: str
    split: str = "test"
    annotation_notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationQuery":
        return cls(
            query_id=data["query_id"],
            query=data["query"],
            relevant_unit_ids=list(data.get("relevant_unit_ids", [])),
            task_category=data.get("task_category", "general"),
            split=data.get("split", "test"),
            annotation_notes=data.get("annotation_notes", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryEvaluationResult:
    """Per-query retrieval evaluation metrics."""
    query_id: str
    query: str
    task_category: str
    split: str
    relevant_ids: List[str]
    retrieved_ids: List[str]
    missing_ids: List[str]
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateEvaluationResult:
    """Aggregate benchmark results across all evaluated queries."""
    total_queries: int
    evaluated_queries: int
    unannotated_queries: int
    mean_recall_at_1: float
    mean_recall_at_5: float
    mean_recall_at_10: float
    mrr: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missing_relevant_ids: Set[str] = field(default_factory=set)
    confidence_intervals_95: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["missing_relevant_ids"] = sorted(list(self.missing_relevant_ids))
        return d

    def summary_table(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"CodeSage Benchmark Evaluation Summary",
            f"{'=' * 60}",
            f"Total Queries        : {self.total_queries}",
            f"Evaluated Queries    : {self.evaluated_queries}",
            f"Unannotated Queries  : {self.unannotated_queries} (handled separately)",
            f"{'-' * 60}",
            f"Recall@1             : {self.mean_recall_at_1:.4f}",
            f"Recall@5             : {self.mean_recall_at_5:.4f}",
            f"Recall@10            : {self.mean_recall_at_10:.4f}",
            f"MRR (Mean Recip Rank): {self.mrr:.4f}",
            f"nDCG@5               : {self.mean_ndcg_at_5:.4f}",
            f"nDCG@10              : {self.mean_ndcg_at_10:.4f}",
            f"{'-' * 60}",
        ]

        if self.missing_relevant_ids:
            lines.append(f"WARNING: Missing Relevant IDs in Index ({len(self.missing_relevant_ids)}):")
            for mid in sorted(self.missing_relevant_ids):
                lines.append(f"  - {mid}")
            lines.append(f"{'-' * 60}")

        if self.category_metrics:
            lines.append("Per-Category Breakdown (Recall@5 / MRR / nDCG@5):")
            for cat, metrics in sorted(self.category_metrics.items()):
                lines.append(
                    f"  * {cat:<24} | Qs: {int(metrics['count']):2d} | "
                    f"R@5: {metrics['recall_at_5']:.3f} | "
                    f"MRR: {metrics['mrr']:.3f} | "
                    f"nDCG@5: {metrics['ndcg_at_5']:.3f}"
                )
            lines.append(f"{'=' * 60}")

        return "\n".join(lines)


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculate Recall@K: proportion of unique relevant units retrieved in top-k.
    
    Safely deduplicates retrieved_ids preserving rank order, and guards against empty relevance sets.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0
    seen = set()
    deduped = []
    for uid in retrieved_ids:
        if uid not in seen:
            seen.add(uid)
            deduped.append(uid)
    top_k = deduped[:k]
    hits = sum(1 for uid in top_k if uid in relevant_ids)
    return hits / float(len(relevant_ids))


def calculate_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant unit.
    
    Safely deduplicates retrieved_ids preserving rank order.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    seen = set()
    rank = 1
    for uid in retrieved_ids:
        if uid in seen:
            continue
        seen.add(uid)
        if uid in relevant_ids:
            return 1.0 / rank
        rank += 1
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank K using binary relevance.
    
    Safely deduplicates retrieved_ids preserving rank order.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    seen = set()
    deduped = []
    for uid in retrieved_ids:
        if uid not in seen:
            seen.add(uid)
            deduped.append(uid)

    # DCG@K
    dcg = 0.0
    for rank, uid in enumerate(deduped[:k], start=1):
        if uid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG@K (all relevant items ranked at top)
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def load_evaluation_queries(queries_path: Path | str) -> List[EvaluationQuery]:
    """Load evaluation queries from JSONL file."""
    path = Path(queries_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    queries: List[EvaluationQuery] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            queries.append(EvaluationQuery.from_dict(data))
    return queries


def compute_bootstrap_ci(
    values: List[float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for mean metric."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    boot_means = [float(np.mean(rng.choice(arr, size=n, replace=True))) for _ in range(n_bootstraps)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return (round(low, 4), round(high, 4))


def evaluate_dataset(
    retriever: Any,
    queries: List[EvaluationQuery],
    top_k: int = 10,
    split_filter: Optional[str] = None,
) -> Tuple[List[QueryEvaluationResult], AggregateEvaluationResult]:
    """Evaluate a retriever over a list of EvaluationQuery records."""
    if split_filter and split_filter.lower() != "all":
        queries = [q for q in queries if q.split == split_filter]

    indexed_unit_ids = set(retriever.unit_ids) if hasattr(retriever, "unit_ids") else set()

    query_results: List[QueryEvaluationResult] = []
    all_missing_ids: Set[str] = set()

    evaluated_queries: List[QueryEvaluationResult] = []
    unannotated_count = 0

    category_buckets: Dict[str, List[QueryEvaluationResult]] = {}

    for eq in queries:
        rel_set = set(eq.relevant_unit_ids)

        # Track missing relevant unit IDs that do not exist in the index
        missing_ids = [uid for uid in eq.relevant_unit_ids if uid not in indexed_unit_ids]
        all_missing_ids.update(missing_ids)

        # Retrieve top_k results
        search_results = retriever.search(eq.query, top_k=top_k)
        retrieved_ids = [r.unit_id for r in search_results]

        if not rel_set:
            # Query has no relevant units (e.g., negative control or unannotated query)
            unannotated_count += 1
            res = QueryEvaluationResult(
                query_id=eq.query_id,
                query=eq.query,
                task_category=eq.task_category,
                split=eq.split,
                relevant_ids=eq.relevant_unit_ids,
                retrieved_ids=retrieved_ids,
                missing_ids=missing_ids,
                recall_at_1=0.0,
                recall_at_5=0.0,
                recall_at_10=0.0,
                mrr=0.0,
                ndcg_at_5=0.0,
                ndcg_at_10=0.0,
            )
            query_results.append(res)
            continue

        r1 = calculate_recall_at_k(retrieved_ids, rel_set, k=1)
        r5 = calculate_recall_at_k(retrieved_ids, rel_set, k=5)
        r10 = calculate_recall_at_k(retrieved_ids, rel_set, k=10)
        mrr = calculate_mrr(retrieved_ids, rel_set)
        ndcg5 = calculate_ndcg_at_k(retrieved_ids, rel_set, k=5)
        ndcg10 = calculate_ndcg_at_k(retrieved_ids, rel_set, k=10)

        res = QueryEvaluationResult(
            query_id=eq.query_id,
            query=eq.query,
            task_category=eq.task_category,
            split=eq.split,
            relevant_ids=eq.relevant_unit_ids,
            retrieved_ids=retrieved_ids,
            missing_ids=missing_ids,
            recall_at_1=r1,
            recall_at_5=r5,
            recall_at_10=r10,
            mrr=mrr,
            ndcg_at_5=ndcg5,
            ndcg_at_10=ndcg10,
        )
        query_results.append(res)
        evaluated_queries.append(res)

        category_buckets.setdefault(eq.task_category, []).append(res)

    num_eval = len(evaluated_queries)
    if num_eval > 0:
        mean_r1 = sum(q.recall_at_1 for q in evaluated_queries) / num_eval
        mean_r5 = sum(q.recall_at_5 for q in evaluated_queries) / num_eval
        mean_r10 = sum(q.recall_at_10 for q in evaluated_queries) / num_eval
        mean_mrr = sum(q.mrr for q in evaluated_queries) / num_eval
        mean_ndcg5 = sum(q.ndcg_at_5 for q in evaluated_queries) / num_eval
        mean_ndcg10 = sum(q.ndcg_at_10 for q in evaluated_queries) / num_eval

        r1_vals = [q.recall_at_1 for q in evaluated_queries]
        r5_vals = [q.recall_at_5 for q in evaluated_queries]
        r10_vals = [q.recall_at_10 for q in evaluated_queries]
        mrr_vals = [q.mrr for q in evaluated_queries]
        ndcg5_vals = [q.ndcg_at_5 for q in evaluated_queries]
        ndcg10_vals = [q.ndcg_at_10 for q in evaluated_queries]

        ci_map = {
            "recall_at_1": compute_bootstrap_ci(r1_vals),
            "recall_at_5": compute_bootstrap_ci(r5_vals),
            "recall_at_10": compute_bootstrap_ci(r10_vals),
            "mrr": compute_bootstrap_ci(mrr_vals),
            "ndcg_at_5": compute_bootstrap_ci(ndcg5_vals),
            "ndcg_at_10": compute_bootstrap_ci(ndcg10_vals),
        }
    else:
        mean_r1 = mean_r5 = mean_r10 = mean_mrr = mean_ndcg5 = mean_ndcg10 = 0.0
        ci_map = {}

    # Category breakdowns
    category_metrics: Dict[str, Dict[str, float]] = {}
    for cat, cat_results in category_buckets.items():
        cat_len = len(cat_results)
        category_metrics[cat] = {
            "count": float(cat_len),
            "recall_at_1": sum(q.recall_at_1 for q in cat_results) / cat_len,
            "recall_at_5": sum(q.recall_at_5 for q in cat_results) / cat_len,
            "recall_at_10": sum(q.recall_at_10 for q in cat_results) / cat_len,
            "mrr": sum(q.mrr for q in cat_results) / cat_len,
            "ndcg_at_5": sum(q.ndcg_at_5 for q in cat_results) / cat_len,
            "ndcg_at_10": sum(q.ndcg_at_10 for q in cat_results) / cat_len,
        }

    aggregate = AggregateEvaluationResult(
        total_queries=len(queries),
        evaluated_queries=num_eval,
        unannotated_queries=unannotated_count,
        mean_recall_at_1=mean_r1,
        mean_recall_at_5=mean_r5,
        mean_recall_at_10=mean_r10,
        mrr=mean_mrr,
        mean_ndcg_at_5=mean_ndcg5,
        mean_ndcg_at_10=mean_ndcg10,
        category_metrics=category_metrics,
        missing_relevant_ids=all_missing_ids,
        confidence_intervals_95=ci_map,
    )

    return query_results, aggregate


@dataclass
class NegativeControlQueryAnalysis:
    """Per-query analysis of false-positive behavior on irrelevant negative controls."""
    query_id: str
    query: str
    task_category: str
    top_score: float
    top_unit_id: str
    top_rel_path: str
    top_scores: List[float]
    top_unit_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NegativeControlReport:
    """Aggregate analysis of false alarms on negative control queries."""
    total_negative_queries: int
    mean_top1_score: float
    p50_top1_score: float
    max_top1_score: float
    min_top1_score: float
    threshold_exceedance_rates: Dict[str, float]
    calibration_warning: str
    per_query_analyses: List[NegativeControlQueryAnalysis]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_query_analyses"] = [q.to_dict() for q in self.per_query_analyses]
        return d


def evaluate_negative_controls(
    retriever: Any,
    queries: List[EvaluationQuery],
    top_k: int = 5,
    thresholds: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> NegativeControlReport:
    """Rigorously evaluate a retriever against negative control queries.
    
    Negative control queries represent requests that have no implementation in the target codebase.
    This analysis measures how often irrelevant queries produce high-confidence-looking similarity scores,
    and proves that raw vector cosine similarities are uncalibrated measures of relevance.
    """
    neg_queries = [q for q in queries if q.split == "negative_control" or len(q.relevant_unit_ids) == 0]
    if not neg_queries:
        return NegativeControlReport(
            total_negative_queries=0,
            mean_top1_score=0.0,
            p50_top1_score=0.0,
            max_top1_score=0.0,
            min_top1_score=0.0,
            threshold_exceedance_rates={f">={t:.1f}": 0.0 for t in thresholds},
            calibration_warning="No negative control queries found in dataset.",
            per_query_analyses=[],
        )

    analyses: List[NegativeControlQueryAnalysis] = []
    top1_scores: List[float] = []

    for eq in neg_queries:
        search_results = retriever.search(eq.query, top_k=top_k)
        if search_results:
            top_res = search_results[0]
            t_score = round(float(top_res.score), 4)
            t_uid = top_res.unit_id
            t_path = getattr(top_res, "rel_path", "")
            scores = [round(float(r.score), 4) for r in search_results]
            uids = [r.unit_id for r in search_results]
        else:
            t_score = 0.0
            t_uid = "NONE"
            t_path = "NONE"
            scores = []
            uids = []

        top1_scores.append(t_score)
        analyses.append(
            NegativeControlQueryAnalysis(
                query_id=eq.query_id,
                query=eq.query,
                task_category=eq.task_category,
                top_score=t_score,
                top_unit_id=t_uid,
                top_rel_path=t_path,
                top_scores=scores,
                top_unit_ids=uids,
            )
        )

    n_queries = len(top1_scores)
    exceedance_rates: Dict[str, float] = {}
    for t in thresholds:
        count = sum(1 for s in top1_scores if s >= t)
        exceedance_rates[f">={t:.1f}"] = round(count / float(n_queries), 4)

    mean_top1 = float(np.mean(top1_scores)) if top1_scores else 0.0
    p50_top1 = float(np.median(top1_scores)) if top1_scores else 0.0
    max_top1 = float(np.max(top1_scores)) if top1_scores else 0.0
    min_top1 = float(np.min(top1_scores)) if top1_scores else 0.0

    return NegativeControlReport(
        total_negative_queries=n_queries,
        mean_top1_score=round(mean_top1, 4),
        p50_top1_score=round(p50_top1, 4),
        max_top1_score=round(max_top1, 4),
        min_top1_score=round(min_top1, 4),
        threshold_exceedance_rates=exceedance_rates,
        calibration_warning="Raw cosine similarity is an uncalibrated ranking metric, not a posterior probability of relevance.",
        per_query_analyses=analyses,
    )


def export_aggregate_json(aggregate: AggregateEvaluationResult, output_path: Path | str) -> None:
    """Export aggregate evaluation metrics to a JSON file."""
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(aggregate.to_dict(), f, indent=2)


def export_query_results_csv(query_results: List[QueryEvaluationResult], output_path: Path | str) -> None:
    """Export query-level evaluation metrics to a CSV file."""
    path = Path(output_path)
    fieldnames = [
        "query_id",
        "split",
        "task_category",
        "query",
        "num_relevant",
        "num_retrieved",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
        "relevant_ids",
        "retrieved_ids",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qr in query_results:
            writer.writerow({
                "query_id": qr.query_id,
                "split": getattr(qr, "split", "test"),
                "task_category": qr.task_category,
                "query": qr.query,
                "num_relevant": len(qr.relevant_ids),
                "num_retrieved": len(qr.retrieved_ids),
                "recall_at_1": f"{qr.recall_at_1:.4f}",
                "recall_at_5": f"{qr.recall_at_5:.4f}",
                "recall_at_10": f"{qr.recall_at_10:.4f}",
                "mrr": f"{qr.mrr:.4f}",
                "ndcg_at_5": f"{qr.ndcg_at_5:.4f}",
                "ndcg_at_10": f"{qr.ndcg_at_10:.4f}",
                "relevant_ids": ";".join(qr.relevant_ids),
                "retrieved_ids": ";".join(qr.retrieved_ids),
            })


@dataclass
class ComparativeBenchmarkResult:
    """Side-by-side benchmark comparison between two retrievers."""
    tfidf_aggregate: AggregateEvaluationResult
    semantic_aggregate: AggregateEvaluationResult
    tfidf_latencies_ms: List[float]
    semantic_latencies_ms: List[float]
    tfidf_indexing_time_s: float
    semantic_indexing_time_s: float
    paired_significance: Dict[str, Dict[str, Any]]
    model_storage_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tfidf": self.tfidf_aggregate.to_dict(),
            "semantic": self.semantic_aggregate.to_dict(),
            "latency": {
                "tfidf_mean_ms": float(np.mean(self.tfidf_latencies_ms)),
                "tfidf_p50_ms": float(np.median(self.tfidf_latencies_ms)),
                "tfidf_p95_ms": float(np.percentile(self.tfidf_latencies_ms, 95)),
                "semantic_mean_ms": float(np.mean(self.semantic_latencies_ms)),
                "semantic_p50_ms": float(np.median(self.semantic_latencies_ms)),
                "semantic_p95_ms": float(np.percentile(self.semantic_latencies_ms, 95)),
            },
            "indexing_time_seconds": {
                "tfidf": round(self.tfidf_indexing_time_s, 4),
                "semantic": round(self.semantic_indexing_time_s, 4),
            },
            "paired_significance": self.paired_significance,
            "model_storage": self.model_storage_info,
        }

    def summary_table(self) -> str:
        t_agg = self.tfidf_aggregate
        s_agg = self.semantic_aggregate

        t_lat_mean = float(np.mean(self.tfidf_latencies_ms))
        t_lat_p50 = float(np.median(self.tfidf_latencies_ms))
        t_lat_p95 = float(np.percentile(self.tfidf_latencies_ms, 95))

        s_lat_mean = float(np.mean(self.semantic_latencies_ms))
        s_lat_p50 = float(np.median(self.semantic_latencies_ms))
        s_lat_p95 = float(np.percentile(self.semantic_latencies_ms, 95))

        metrics = [
            ("Recall@1", t_agg.mean_recall_at_1, s_agg.mean_recall_at_1),
            ("Recall@5", t_agg.mean_recall_at_5, s_agg.mean_recall_at_5),
            ("Recall@10", t_agg.mean_recall_at_10, s_agg.mean_recall_at_10),
            ("MRR", t_agg.mrr, s_agg.mrr),
            ("nDCG@5", t_agg.mean_ndcg_at_5, s_agg.mean_ndcg_at_5),
            ("nDCG@10", t_agg.mean_ndcg_at_10, s_agg.mean_ndcg_at_10),
        ]

        lines = [
            "=" * 78,
            f"CodeSage Milestone 2 vs Milestone 3: Comparative Benchmark",
            "=" * 78,
            f"Evaluated Queries: {t_agg.evaluated_queries} | Unannotated Negative Controls: {t_agg.unannotated_queries}",
            "-" * 78,
            f"{'Metric':<14} | {'TF-IDF (Lexical)':<18} | {'Semantic (MiniLM)':<18} | {'Delta':<10} | {'p-value':<8}",
            "-" * 78,
        ]

        for name, t_val, s_val in metrics:
            delta = s_val - t_val
            sig_info = self.paired_significance.get(name, {})
            pval_str = f"{sig_info.get('p_value', 1.0):.4f}" if "p_value" in sig_info else "N/A"
            delta_str = f"{delta:+.4f}"
            lines.append(f"{name:<14} | {t_val:<18.4f} | {s_val:<18.4f} | {delta_str:<10} | {pval_str:<8}")

        lines.extend([
            "-" * 78,
            "Latency & Resource Consumption:",
            f"  * Query Latency (Mean)   : TF-IDF: {t_lat_mean:.2f} ms | Semantic: {s_lat_mean:.2f} ms",
            f"  * Query Latency (p50/p95): TF-IDF: {t_lat_p50:.2f} / {t_lat_p95:.2f} ms | Semantic: {s_lat_p50:.2f} / {s_lat_p95:.2f} ms",
            f"  * Indexing / Embed Time  : TF-IDF: {self.tfidf_indexing_time_s:.3f} s | Semantic: {self.semantic_indexing_time_s:.3f} s",
            f"  * Model Storage Footprint: {self.model_storage_info.get('model_size_mb', 0):.1f} MB (weights) + {self.model_storage_info.get('embedding_cache_kb', 0):.1f} KB (embeddings)",
            "=" * 78,
        ])

        return "\n".join(lines)


def run_paired_significance_test(
    tfidf_results: List[QueryEvaluationResult],
    semantic_results: List[QueryEvaluationResult],
) -> Dict[str, Dict[str, Any]]:
    """Compute paired differences and Wilcoxon signed-rank / paired t-test for each metric."""
    try:
        from scipy import stats
    except ImportError:
        stats = None

    metrics = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_5", "ndcg_at_10"]
    metric_names = {
        "recall_at_1": "Recall@1",
        "recall_at_5": "Recall@5",
        "recall_at_10": "Recall@10",
        "mrr": "MRR",
        "ndcg_at_5": "nDCG@5",
        "ndcg_at_10": "nDCG@10",
    }

    # Match queries by query_id and filter unannotated
    paired_data: Dict[str, Tuple[List[float], List[float]]] = {m: ([], []) for m in metrics}

    s_map = {r.query_id: r for r in semantic_results}
    for t_res in tfidf_results:
        if not t_res.relevant_ids:
            continue
        s_res = s_map.get(t_res.query_id)
        if not s_res:
            continue

        for m in metrics:
            t_val = getattr(t_res, m)
            s_val = getattr(s_res, m)
            paired_data[m][0].append(t_val)
            paired_data[m][1].append(s_val)

    significance: Dict[str, Dict[str, Any]] = {}
    n_queries = len(paired_data[metrics[0]][0]) if metrics else 0

    for m in metrics:
        t_vals = np.array(paired_data[m][0])
        s_vals = np.array(paired_data[m][1])
        diffs = s_vals - t_vals
        mean_diff = float(np.mean(diffs))
        ci_diff = compute_bootstrap_ci(list(diffs))

        non_zero = diffs[diffs != 0]
        p_val = 1.0
        test_type = "none"

        if stats is not None and len(non_zero) >= 5:
            try:
                # Wilcoxon signed-rank test on non-zero differences
                res = stats.wilcoxon(s_vals, t_vals, zero_method="pratt")
                p_val = float(res.pvalue)
                test_type = "wilcoxon_signed_rank"
            except Exception:
                try:
                    res = stats.ttest_rel(s_vals, t_vals)
                    p_val = float(res.pvalue)
                    test_type = "paired_ttest"
                except Exception:
                    p_val = 1.0
        elif stats is not None and len(non_zero) > 0:
            try:
                res = stats.ttest_rel(s_vals, t_vals)
                p_val = float(res.pvalue)
                test_type = "paired_ttest"
            except Exception:
                p_val = 1.0

        label = metric_names[m]
        significance[label] = {
            "n_queries": n_queries,
            "mean_difference": round(mean_diff, 4),
            "ci_95_diff": ci_diff,
            "p_value": round(p_val, 6),
            "test_type": test_type,
            "statistically_significant": bool(p_val < 0.05),
            "sample_size_limitation": "Sample size N < 30 limits statistical power; non-significance does not establish equivalence." if n_queries < 30 else "",
        }

    return significance


def export_comparative_csv(
    tfidf_results: List[QueryEvaluationResult],
    semantic_results: List[QueryEvaluationResult],
    output_path: Path | str,
) -> None:
    """Export query-level side-by-side comparison to CSV."""
    path = Path(output_path)
    s_map = {r.query_id: r for r in semantic_results}

    fieldnames = [
        "query_id",
        "task_category",
        "query",
        "tfidf_r1",
        "semantic_r1",
        "delta_r1",
        "tfidf_r5",
        "semantic_r5",
        "delta_r5",
        "tfidf_mrr",
        "semantic_mrr",
        "delta_mrr",
        "tfidf_ndcg5",
        "semantic_ndcg5",
        "delta_ndcg5",
        "tfidf_retrieved_top3",
        "semantic_retrieved_top3",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tfidf_results:
            s = s_map.get(t.query_id)
            if not s:
                continue

            writer.writerow({
                "query_id": t.query_id,
                "task_category": t.task_category,
                "query": t.query,
                "tfidf_r1": f"{t.recall_at_1:.4f}",
                "semantic_r1": f"{s.recall_at_1:.4f}",
                "delta_r1": f"{s.recall_at_1 - t.recall_at_1:+.4f}",
                "tfidf_r5": f"{t.recall_at_5:.4f}",
                "semantic_r5": f"{s.recall_at_5:.4f}",
                "delta_r5": f"{s.recall_at_5 - t.recall_at_5:+.4f}",
                "tfidf_mrr": f"{t.mrr:.4f}",
                "semantic_mrr": f"{s.mrr:.4f}",
                "delta_mrr": f"{s.mrr - t.mrr:+.4f}",
                "tfidf_ndcg5": f"{t.ndcg_at_5:.4f}",
                "semantic_ndcg5": f"{s.ndcg_at_5:.4f}",
                "delta_ndcg5": f"{s.ndcg_at_5 - t.ndcg_at_5:+.4f}",
                "tfidf_retrieved_top3": ";".join(t.retrieved_ids[:3]),
                "semantic_retrieved_top3": ";".join(s.retrieved_ids[:3]),
            })


@dataclass
class ThreeWayBenchmarkResult:
    """Side-by-side benchmark comparison covering all 3 retrieval regimes."""
    tfidf_aggregate: AggregateEvaluationResult
    semantic_aggregate: AggregateEvaluationResult
    code_aggregate: AggregateEvaluationResult
    tfidf_latencies_ms: List[float]
    semantic_latencies_ms: List[float]
    code_latencies_ms: List[float]
    tfidf_indexing_time_s: float
    semantic_indexing_time_s: float
    code_indexing_time_s: float
    paired_significance: Dict[str, Dict[str, Any]]
    model_storage_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tfidf": self.tfidf_aggregate.to_dict(),
            "semantic": self.semantic_aggregate.to_dict(),
            "code_transformer": self.code_aggregate.to_dict(),
            "latency": {
                "tfidf_mean_ms": float(np.mean(self.tfidf_latencies_ms)),
                "tfidf_p50_ms": float(np.median(self.tfidf_latencies_ms)),
                "tfidf_p95_ms": float(np.percentile(self.tfidf_latencies_ms, 95)),
                "semantic_mean_ms": float(np.mean(self.semantic_latencies_ms)),
                "semantic_p50_ms": float(np.median(self.semantic_latencies_ms)),
                "semantic_p95_ms": float(np.percentile(self.semantic_latencies_ms, 95)),
                "code_mean_ms": float(np.mean(self.code_latencies_ms)),
                "code_p50_ms": float(np.median(self.code_latencies_ms)),
                "code_p95_ms": float(np.percentile(self.code_latencies_ms, 95)),
            },
            "indexing_time_seconds": {
                "tfidf": round(self.tfidf_indexing_time_s, 4),
                "semantic": round(self.semantic_indexing_time_s, 4),
                "code_transformer": round(self.code_indexing_time_s, 4),
            },
            "paired_significance": self.paired_significance,
            "model_storage": self.model_storage_info,
        }

    def summary_table(self) -> str:
        t = self.tfidf_aggregate
        s = self.semantic_aggregate
        c = self.code_aggregate

        metrics = [
            ("Recall@1", t.mean_recall_at_1, s.mean_recall_at_1, c.mean_recall_at_1),
            ("Recall@5", t.mean_recall_at_5, s.mean_recall_at_5, c.mean_recall_at_5),
            ("Recall@10", t.mean_recall_at_10, s.mean_recall_at_10, c.mean_recall_at_10),
            ("MRR", t.mrr, s.mrr, c.mrr),
            ("nDCG@5", t.mean_ndcg_at_5, s.mean_ndcg_at_5, c.mean_ndcg_at_5),
            ("nDCG@10", t.mean_ndcg_at_10, s.mean_ndcg_at_10, c.mean_ndcg_at_10),
        ]

        lines = [
            "=" * 92,
            "CodeSage Retrieval Benchmarks: Lexical vs General Semantic vs Code Transformer",
            "=" * 92,
            f"Evaluated Queries: {t.evaluated_queries} | Unannotated Controls: {t.unannotated_queries}",
            "-" * 92,
            f"{'Metric':<12} | {'TF-IDF (Lexical)':<18} | {'Semantic (MiniLM)':<18} | {'Code (DistilRoBERTa)':<21} | {'Delta (Code vs TF-IDF)'}",
            "-" * 92,
        ]

        for name, t_val, s_val, c_val in metrics:
            delta_code_tfidf = c_val - t_val
            lines.append(
                f"{name:<12} | {t_val:<18.4f} | {s_val:<18.4f} | {c_val:<21.4f} | {delta_code_tfidf:+.4f}"
            )

        t_mean = float(np.mean(self.tfidf_latencies_ms))
        s_mean = float(np.mean(self.semantic_latencies_ms))
        c_mean = float(np.mean(self.code_latencies_ms))
        t_p50 = float(np.median(self.tfidf_latencies_ms))
        s_p50 = float(np.median(self.semantic_latencies_ms))
        c_p50 = float(np.median(self.code_latencies_ms))

        lines.extend([
            "-" * 92,
            "Latency & Computation Profiles:",
            f"  * Query Latency (Mean)   : TF-IDF: {t_mean:.2f} ms | Semantic: {s_mean:.2f} ms | Code: {c_mean:.2f} ms",
            f"  * Query Latency (p50)    : TF-IDF: {t_p50:.2f} ms | Semantic: {s_p50:.2f} ms | Code: {c_p50:.2f} ms",
            f"  * Indexing Time (s)      : TF-IDF: {self.tfidf_indexing_time_s:.3f} s | Semantic: {self.semantic_indexing_time_s:.3f} s | Code: {self.code_indexing_time_s:.3f} s",
            f"  * Model Weight Footprint : TF-IDF: 0 MB | Semantic: 86.7 MB | Code: 313.3 MB",
            "=" * 92,
        ])

        # Category Breakdown for Code Transformer vs TF-IDF
        lines.append("\nPer-Category Recall@5 Comparison (TF-IDF vs Semantic vs Code):")
        all_cats = sorted(set(list(t.category_metrics.keys()) + list(c.category_metrics.keys())))
        for cat in all_cats:
            t_r5 = t.category_metrics.get(cat, {}).get("recall_at_5", 0.0)
            s_r5 = s.category_metrics.get(cat, {}).get("recall_at_5", 0.0)
            c_r5 = c.category_metrics.get(cat, {}).get("recall_at_5", 0.0)
            count = int(t.category_metrics.get(cat, {}).get("count", 0))
            lines.append(f"  * {cat:<24} (n={count:1d}) | TF-IDF: {t_r5:.3f} | Semantic: {s_r5:.3f} | Code: {c_r5:.3f}")
        lines.append("=" * 92)

        return "\n".join(lines)


def export_three_way_csv(
    tfidf_results: List[QueryEvaluationResult],
    semantic_results: List[QueryEvaluationResult],
    code_results: List[QueryEvaluationResult],
    output_path: Path | str,
) -> None:
    """Export query-level three-way comparison to CSV."""
    path = Path(output_path)
    s_map = {r.query_id: r for r in semantic_results}
    c_map = {r.query_id: r for r in code_results}

    fieldnames = [
        "query_id",
        "task_category",
        "query",
        "tfidf_r1",
        "semantic_r1",
        "code_r1",
        "tfidf_r5",
        "semantic_r5",
        "code_r5",
        "tfidf_mrr",
        "semantic_mrr",
        "code_mrr",
        "tfidf_ndcg5",
        "semantic_ndcg5",
        "code_ndcg5",
        "tfidf_top3",
        "semantic_top3",
        "code_top3",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tfidf_results:
            s = s_map.get(t.query_id)
            c = c_map.get(t.query_id)
            if not s or not c:
                continue

            writer.writerow({
                "query_id": t.query_id,
                "task_category": t.task_category,
                "query": t.query,
                "tfidf_r1": f"{t.recall_at_1:.4f}",
                "semantic_r1": f"{s.recall_at_1:.4f}",
                "code_r1": f"{c.recall_at_1:.4f}",
                "tfidf_r5": f"{t.recall_at_5:.4f}",
                "semantic_r5": f"{s.recall_at_5:.4f}",
                "code_r5": f"{c.recall_at_5:.4f}",
                "tfidf_mrr": f"{t.mrr:.4f}",
                "semantic_mrr": f"{s.mrr:.4f}",
                "code_mrr": f"{c.mrr:.4f}",
                "tfidf_ndcg5": f"{t.ndcg_at_5:.4f}",
                "semantic_ndcg5": f"{s.ndcg_at_5:.4f}",
                "code_ndcg5": f"{c.ndcg_at_5:.4f}",
                "tfidf_top3": ";".join(t.retrieved_ids[:3]),
                "semantic_top3": ";".join(s.retrieved_ids[:3]),
                "code_top3": ";".join(c.retrieved_ids[:3]),
            })


