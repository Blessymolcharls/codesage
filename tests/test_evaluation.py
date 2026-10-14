"""Automated unit and integration tests for CodeSage evaluation metrics and harness."""

import json
from pathlib import Path
import pytest

from codesage.evaluation import (
    EvaluationQuery,
    QueryEvaluationResult,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    evaluate_dataset,
    export_aggregate_json,
    export_query_results_csv,
    load_evaluation_queries,
)
from codesage.retrieval import TfidfRetriever, SearchResult


def test_metric_calculations_known_examples():
    """Verify Recall, MRR, and nDCG calculations against hand-computed ground truth."""
    # Relevant units are {'u1', 'u2'}
    relevant_ids = {"u1", "u2"}

    # Scenario A: Perfect ranking -> [u1, u2, u3, u4, u5]
    retrieved_a = ["u1", "u2", "u3", "u4", "u5"]
    assert calculate_recall_at_k(retrieved_a, relevant_ids, k=1) == 0.5  # 1/2
    assert calculate_recall_at_k(retrieved_a, relevant_ids, k=2) == 1.0  # 2/2
    assert calculate_recall_at_k(retrieved_a, relevant_ids, k=5) == 1.0
    assert calculate_mrr(retrieved_a, relevant_ids) == 1.0  # rank 1 -> 1/1 = 1.0

    # IDCG@5 for 2 items: 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
    # DCG@5 is identical -> nDCG = 1.0
    assert pytest.approx(calculate_ndcg_at_k(retrieved_a, relevant_ids, k=5), 0.001) == 1.0

    # Scenario B: Second and fourth positions -> [u3, u1, u4, u2, u5]
    retrieved_b = ["u3", "u1", "u4", "u2", "u5"]
    assert calculate_recall_at_k(retrieved_b, relevant_ids, k=1) == 0.0
    assert calculate_recall_at_k(retrieved_b, relevant_ids, k=2) == 0.5
    assert calculate_recall_at_k(retrieved_b, relevant_ids, k=5) == 1.0
    assert pytest.approx(calculate_mrr(retrieved_b, relevant_ids), 0.001) == 0.5  # 1st hit at rank 2 -> 1/2 = 0.5

    # DCG@5: rank 2 gives 1/log2(3) = 0.63092975; rank 4 gives 1/log2(5) = 0.43067655
    # sum = 1.0616063; IDCG = 1.63092975 -> nDCG@5 = 1.0616063 / 1.63092975 = 0.6509
    expected_ndcg = (1.0 / 1.5849625 + 1.0 / 2.321928) / (1.0 / 1.0 + 1.0 / 1.5849625)
    assert pytest.approx(calculate_ndcg_at_k(retrieved_b, relevant_ids, k=5), 0.001) == pytest.approx(expected_ndcg, 0.001)

    # Scenario C: No relevant items retrieved
    retrieved_c = ["x", "y", "z"]
    assert calculate_recall_at_k(retrieved_c, relevant_ids, k=1) == 0.0
    assert calculate_recall_at_k(retrieved_c, relevant_ids, k=5) == 0.0
    assert calculate_mrr(retrieved_c, relevant_ids) == 0.0
    assert calculate_ndcg_at_k(retrieved_c, relevant_ids, k=5) == 0.0


def test_missing_relevant_ids_detected(tmp_path: Path):
    """Evaluation flags relevant IDs that do not exist in the index."""
    units = [
        {
            "unit_id": "existing_1",
            "rel_path": "a.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "foo",
            "start_line": 1,
            "end_line": 2,
            "source_code": "function foo() {}",
            "extraction_method": "fallback",
        }
    ]
    retriever = TfidfRetriever()
    retriever.fit(units)

    queries = [
        EvaluationQuery(
            query_id="q1",
            query="test foo",
            relevant_unit_ids=["existing_1", "ghost_nonexistent_id"],
            task_category="test",
        )
    ]

    query_results, aggregate = evaluate_dataset(retriever, queries, top_k=5)
    assert aggregate.missing_relevant_ids == {"ghost_nonexistent_id"}
    assert len(query_results) == 1
    assert "ghost_nonexistent_id" in query_results[0].missing_ids


def test_unannotated_queries_handled_separately():
    """Queries with 0 relevant IDs are reported separately and do not distort recall."""
    retriever = TfidfRetriever()
    retriever.fit([
        {
            "unit_id": "u1",
            "rel_path": "a.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "foo",
            "start_line": 1,
            "end_line": 2,
            "source_code": "function foo() {}",
            "extraction_method": "fallback",
        }
    ])

    queries = [
        EvaluationQuery(
            query_id="q_valid",
            query="foo",
            relevant_unit_ids=["u1"],
            task_category="category_a",
        ),
        EvaluationQuery(
            query_id="q_unannotated",
            query="completely unsupported feature",
            relevant_unit_ids=[],
            task_category="negative_control",
        ),
    ]

    query_results, aggregate = evaluate_dataset(retriever, queries, top_k=5)
    assert aggregate.total_queries == 2
    assert aggregate.evaluated_queries == 1
    assert aggregate.unannotated_queries == 1
    assert aggregate.mean_recall_at_1 == 1.0  # Only q_valid evaluated


def test_end_to_end_evaluation_pipeline(tmp_path: Path):
    """End-to-end evaluation pipeline with synthetic index, queries, and JSON/CSV export."""
    units = [
        {
            "unit_id": "u_pdf",
            "rel_path": "pdf.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "parsePdf",
            "start_line": 1,
            "end_line": 5,
            "source_code": "function parsePdf() { return 'pdf'; }",
            "extraction_method": "tree_sitter",
        },
        {
            "unit_id": "u_auth",
            "rel_path": "auth.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "authenticateUser",
            "start_line": 1,
            "end_line": 5,
            "source_code": "function authenticateUser() { return 'token'; }",
            "extraction_method": "tree_sitter",
        },
    ]

    index_file = tmp_path / "units.jsonl"
    with open(index_file, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")

    queries_file = tmp_path / "queries.jsonl"
    queries_data = [
        {
            "query_id": "q1",
            "query": "parse PDF document",
            "relevant_unit_ids": ["u_pdf"],
            "task_category": "pdf_processing",
            "annotation_notes": "synthetic test query",
        },
        {
            "query_id": "q2",
            "query": "authenticate user",
            "relevant_unit_ids": ["u_auth"],
            "task_category": "auth",
            "annotation_notes": "synthetic test query",
        },
    ]
    with open(queries_file, "w", encoding="utf-8") as f:
        for q in queries_data:
            f.write(json.dumps(q) + "\n")

    queries = load_evaluation_queries(queries_file)
    retriever = TfidfRetriever()
    retriever.fit_from_jsonl(index_file)

    query_results, aggregate = evaluate_dataset(retriever, queries, top_k=5)
    assert aggregate.evaluated_queries == 2
    assert aggregate.mean_recall_at_1 == 1.0
    assert aggregate.mrr == 1.0
    assert aggregate.mean_ndcg_at_5 == 1.0

    # Test JSON and CSV exports
    json_export = tmp_path / "results.json"
    csv_export = tmp_path / "results.csv"

    export_aggregate_json(aggregate, json_export)
    export_query_results_csv(query_results, csv_export)

    assert json_export.exists()
    assert csv_export.exists()

    # Verify JSON content
    with open(json_export, "r", encoding="utf-8") as f:
        exported_data = json.load(f)
        assert exported_data["evaluated_queries"] == 2
        assert exported_data["mrr"] == 1.0

    # Verify CSV lines
    csv_content = csv_export.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_content) == 3  # Header + 2 queries
    assert "query_id,split,task_category,query" in csv_content[0]


def test_duplicate_results_deduplication():
    """Verify duplicate IDs in retrieved list do not artificially inflate recall or nDCG."""
    relevant_ids = {"u1", "u2"}

    # Duplicate u1 in top-3: ["u1", "u1", "u2"]
    # Without deduplication, recall would be 3/2 = 1.5!
    # With deduplication, top-2 unique are [u1, u2], recall@2 = 1.0, recall@1 = 0.5
    retrieved_with_dups = ["u1", "u1", "u2", "u3"]
    assert calculate_recall_at_k(retrieved_with_dups, relevant_ids, k=1) == 0.5
    assert calculate_recall_at_k(retrieved_with_dups, relevant_ids, k=2) == 1.0
    assert calculate_mrr(retrieved_with_dups, relevant_ids) == 1.0
    # First unique hit is at rank 1, second unique hit is at rank 2
    assert pytest.approx(calculate_ndcg_at_k(retrieved_with_dups, relevant_ids, k=2), 0.001) == 1.0


def test_empty_relevance_sets():
    """Verify empty relevance sets safely yield 0.0 without divide-by-zero or crashes."""
    empty_rel: set = set()
    retrieved = ["u1", "u2", "u3"]

    assert calculate_recall_at_k(retrieved, empty_rel, k=1) == 0.0
    assert calculate_recall_at_k(retrieved, empty_rel, k=5) == 0.0
    assert calculate_mrr(retrieved, empty_rel) == 0.0
    assert calculate_ndcg_at_k(retrieved, empty_rel, k=5) == 0.0

    # Also test empty retrieved list with non-empty relevant
    assert calculate_recall_at_k([], {"u1"}, k=5) == 0.0
    assert calculate_mrr([], {"u1"}) == 0.0
    assert calculate_ndcg_at_k([], {"u1"}, k=5) == 0.0

    # k <= 0
    assert calculate_recall_at_k(retrieved, {"u1"}, k=0) == 0.0
    assert calculate_ndcg_at_k(retrieved, {"u1"}, k=0) == 0.0


def test_negative_control_evaluation_harness():
    """Verify negative control evaluation reports false alarms, top scores, and calibration warnings."""
    from codesage.evaluation import evaluate_negative_controls

    class MockNegativeRetriever:
        def search(self, query: str, top_k: int = 5):
            # Returns fixed uncalibrated scores
            return [
                SearchResult(
                    unit_id="doc1",
                    score=0.45,
                    rel_path="doc1.js",
                    language="javascript",
                    symbol_name="doc1",
                    unit_type="function",
                    start_line=1,
                    end_line=10,
                    extraction_method="tree_sitter",
                    preview="code",
                    source_code="code",
                ),
                SearchResult(
                    unit_id="doc2",
                    score=0.32,
                    rel_path="doc2.js",
                    language="javascript",
                    symbol_name="doc2",
                    unit_type="function",
                    start_line=1,
                    end_line=10,
                    extraction_method="tree_sitter",
                    preview="code",
                    source_code="code",
                ),
            ]

    neg_queries = [
        EvaluationQuery(
            query_id="neg1",
            query="unsupported feature query 1",
            relevant_unit_ids=[],
            task_category="unsupported",
            split="negative_control",
        ),
        EvaluationQuery(
            query_id="neg2",
            query="unsupported feature query 2",
            relevant_unit_ids=[],
            task_category="unsupported",
            split="negative_control",
        ),
    ]

    report = evaluate_negative_controls(MockNegativeRetriever(), neg_queries, top_k=5)
    assert report.total_negative_queries == 2
    assert report.mean_top1_score == 0.45
    assert report.max_top1_score == 0.45
    assert report.threshold_exceedance_rates[">=0.4"] == 1.0
    assert report.threshold_exceedance_rates[">=0.5"] == 0.0
    assert len(report.per_query_analyses) == 2
    assert "uncalibrated" in report.calibration_warning.lower()


def test_evaluation_split_isolation_and_disjointness():
    """Verify that all evaluation queries in the benchmark have strictly isolated, disjoint splits."""
    root = Path(__file__).resolve().parent.parent
    queries_path = root / "data" / "evaluation_queries.jsonl"
    queries = load_evaluation_queries(queries_path)

    dev_ids = {q.query_id for q in queries if q.split == "dev"}
    test_ids = {q.query_id for q in queries if q.split == "test"}
    neg_ids = {q.query_id for q in queries if q.split == "negative_control"}

    # Disjoint ID sets
    assert len(dev_ids.intersection(test_ids)) == 0, "Dev and Test query IDs overlap!"
    assert len(dev_ids.intersection(neg_ids)) == 0, "Dev and Negative query IDs overlap!"
    assert len(test_ids.intersection(neg_ids)) == 0, "Test and Negative query IDs overlap!"

    # Disjoint normalized query text
    dev_texts = {q.query.lower().strip() for q in queries if q.split == "dev"}
    test_texts = {q.query.lower().strip() for q in queries if q.split == "test"}
    neg_texts = {q.query.lower().strip() for q in queries if q.split == "negative_control"}

    assert len(dev_texts.intersection(test_texts)) == 0, "Dev and Test query texts overlap!"
    assert len(dev_texts.intersection(neg_texts)) == 0, "Dev and Negative query texts overlap!"
    assert len(test_texts.intersection(neg_texts)) == 0, "Test and Negative query texts overlap!"

