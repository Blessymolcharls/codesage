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
