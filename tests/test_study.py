"""Tests for CodeSage Milestone 5: Research Evaluation, Audit, and Ablations."""

import json
from pathlib import Path

import numpy as np
import pytest

from codesage.code_semantic import (
    CodeModelConfig,
    CodeTransformerRetriever,
    compute_code_units_fingerprint,
    prepare_code_search_document,
)
from codesage.evaluation import (
    EvaluationQuery,
    compute_bootstrap_ci,
    evaluate_dataset,
    load_evaluation_queries,
)
from codesage.retrieval import TfidfRetriever, load_code_units_from_jsonl
from codesage.semantic import (
    SemanticModelConfig,
    SemanticRetriever,
    compute_units_fingerprint,
    prepare_semantic_document,
)
from codesage.tokenization import prepare_search_document


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_dataset_audit_and_split_isolation(project_root: Path) -> None:
    """Audit that all evaluation queries have valid splits, zero dev/test overlap, and grounded relevance."""
    queries_path = project_root / "data" / "evaluation_queries.jsonl"
    index_path = project_root / "data" / "code_units.jsonl"

    queries = load_evaluation_queries(queries_path)
    indexed_units = load_code_units_from_jsonl(index_path)
    indexed_ids = {u["unit_id"] for u in indexed_units}

    assert len(queries) >= 30, f"Expected at least 30 queries, got {len(queries)}"

    dev_queries = [q for q in queries if q.split == "dev"]
    test_queries = [q for q in queries if q.split == "test"]
    neg_queries = [q for q in queries if q.split == "negative_control"]

    assert len(dev_queries) > 0, "Dev set must not be empty"
    assert len(test_queries) > 0, "Test set must not be empty"
    assert len(neg_queries) > 0, "Negative control queries must exist"

    # Zero overlap check
    dev_texts = {q.query.lower().strip() for q in dev_queries}
    test_texts = {q.query.lower().strip() for q in test_queries}
    overlap = dev_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Found overlapping queries between dev and test: {overlap}"

    # Check grounding: all positive relevance labels must exist in index
    for q in queries:
        for rid in q.relevant_unit_ids:
            assert rid in indexed_ids, f"Query '{q.query_id}' references unindexed unit: '{rid}'"

    # Negative controls must have empty relevance lists
    for nq in neg_queries:
        assert len(nq.relevant_unit_ids) == 0, f"Negative control query '{nq.query_id}' must have zero relevant units"


def test_metadata_ablation_document_formatting() -> None:
    """Verify that include_metadata=False excludes identifiers and paths across all representations."""
    symbol = "mergePdfs"
    rel_path = "backend/controllers/pdfController.js"
    source = "const mergePdfs = async (req, res) => { return res.json({}); };"

    # 1. Lexical document representation
    doc_with_meta = prepare_search_document(
        symbol_name=symbol,
        rel_path=rel_path,
        source_code=source,
        language="javascript",
        unit_type="function",
        include_metadata=True,
    )
    doc_no_meta = prepare_search_document(
        symbol_name=symbol,
        rel_path=rel_path,
        source_code=source,
        language="javascript",
        unit_type="function",
        include_metadata=False,
    )
    assert "backend controllers pdfController js" in doc_with_meta
    assert doc_no_meta == source

    # 2. Semantic representation
    sem_with_meta = prepare_semantic_document(
        symbol_name=symbol,
        rel_path=rel_path,
        source_code=source,
        language="javascript",
        unit_type="function",
        include_metadata=True,
    )
    sem_no_meta = prepare_semantic_document(
        symbol_name=symbol,
        rel_path=rel_path,
        source_code=source,
        language="javascript",
        unit_type="function",
        include_metadata=False,
    )
    assert "File: backend/controllers/pdfController.js" in sem_with_meta
    assert "File:" not in sem_no_meta
    assert sem_no_meta == source

    # 3. Code Transformer representation
    code_with_meta = prepare_code_search_document(
        symbol_name=symbol,
        rel_path=rel_path,
        source_code=source,
        language="javascript",
        unit_type="function",
        include_metadata=True,
    )
    code_no_meta = prepare_code_search_document(
        symbol_name=symbol,
        rel_path=rel_path,
        source_code=source,
        language="javascript",
        unit_type="function",
        include_metadata=False,
    )
    assert "// [javascript] function mergePdfs" in code_with_meta
    assert "//" not in code_no_meta
    assert code_no_meta == source


def test_cache_fingerprint_invalidation_on_metadata_toggle() -> None:
    """Verify cache fingerprints strictly isolate metadata-included vs ablated embedding caches."""
    dummy_units = [{
        "unit_id": "test:fn",
        "content_sha256": "abc123sha",
        "symbol_name": "fn",
        "repository_relative_path": "test.js",
        "source_text": "function fn() {}",
    }]

    # Semantic fingerprints
    cfg_sem = SemanticModelConfig()
    fp_sem_meta = compute_units_fingerprint(dummy_units, cfg_sem, include_metadata=True)
    fp_sem_nometa = compute_units_fingerprint(dummy_units, cfg_sem, include_metadata=False)
    assert fp_sem_meta != fp_sem_nometa

    # Code Transformer fingerprints
    cfg_code = CodeModelConfig()
    fp_code_meta = compute_code_units_fingerprint(dummy_units, cfg_code, include_metadata=True)
    fp_code_nometa = compute_code_units_fingerprint(dummy_units, cfg_code, include_metadata=False)
    assert fp_code_meta != fp_code_nometa


def test_bootstrap_confidence_interval_math() -> None:
    """Test non-parametric bootstrap confidence interval estimation."""
    values = [0.8, 0.9, 0.85, 0.95, 0.7, 0.88, 0.92, 0.81]
    mean_val = float(np.mean(values))

    low, high = compute_bootstrap_ci(values, n_bootstraps=500, ci=0.95)
    assert low <= mean_val <= high
    assert 0.6 <= low <= 0.85
    assert 0.85 <= high <= 1.0

    # Empty list edge case
    assert compute_bootstrap_ci([]) == (0.0, 0.0)


def test_split_filtering_in_evaluation(project_root: Path) -> None:
    """Verify evaluate_dataset correctly filters queries by split."""
    queries_path = project_root / "data" / "evaluation_queries.jsonl"
    index_path = project_root / "data" / "code_units.jsonl"

    queries = load_evaluation_queries(queries_path)
    retriever = TfidfRetriever()
    retriever.fit_from_jsonl(index_path)

    # Evaluate dev split only
    dev_results, dev_agg = evaluate_dataset(retriever, queries, split_filter="dev")
    assert dev_agg.total_queries == 18
    assert all(qr.split == "dev" for qr in dev_results)

    # Evaluate test split only
    test_results, test_agg = evaluate_dataset(retriever, queries, split_filter="test")
    assert test_agg.total_queries == 18
    assert all(qr.split == "test" for qr in test_results)

    # Both must report 95% confidence intervals
    assert "mrr" in dev_agg.confidence_intervals_95
    assert "recall_at_1" in test_agg.confidence_intervals_95
