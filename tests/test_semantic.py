"""Automated tests for CodeSage general-purpose semantic retrieval."""

import json
from pathlib import Path
import pytest
import numpy as np

from codesage.evaluation import EvaluationQuery, evaluate_dataset
from codesage.semantic import (
    DEFAULT_MODEL_NAME,
    DEFAULT_EMBEDDING_DIM,
    SemanticRetriever,
    SemanticModelConfig,
    compute_units_fingerprint,
    prepare_semantic_document,
)


def test_prepare_semantic_document():
    """Verify document preparation formats header, path, and bounds source code."""
    doc = prepare_semantic_document(
        symbol_name="splitPdf",
        rel_path="backend/controllers/pdfController.js",
        source_code="const splitPdf = async (req, res) => { /* split logic */ };",
        language="javascript",
        unit_type="function",
        max_chars=50,
    )
    assert "[javascript] function splitPdf" in doc
    assert "File: backend/controllers/pdfController.js" in doc
    assert "const splitPdf" in doc
    # Max chars bounds code portion
    assert len(doc.split("File: backend/controllers/pdfController.js\n")[1]) <= 50


def test_empty_inputs(tmp_path: Path):
    """Empty queries, whitespace queries, or empty indexes safely return empty results."""
    retriever = SemanticRetriever(cache_dir=tmp_path)
    assert retriever.search("query") == []

    # Fit with single unit
    retriever.fit([
        {
            "unit_id": "u1",
            "rel_path": "a.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "foo",
            "source_code": "function foo() {}",
            "content_sha256": "abc1",
        }
    ])

    assert retriever.search("") == []
    assert retriever.search("    ") == []
    assert retriever.search("\t\n") == []


def test_stable_mappings_and_vector_dimensions(tmp_path: Path):
    """Verify vectors have correct 384 dimensions and stably map to unit IDs."""
    units = [
        {
            "unit_id": "unit_alpha",
            "rel_path": "alpha.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "alphaFunction",
            "source_code": "function alphaFunction() { return 1; }",
            "content_sha256": "sha_alpha",
        },
        {
            "unit_id": "unit_beta",
            "rel_path": "beta.dart",
            "language": "dart",
            "unit_type": "class",
            "symbol_name": "BetaState",
            "source_code": "class BetaState {}",
            "content_sha256": "sha_beta",
        },
    ]

    retriever = SemanticRetriever(cache_dir=tmp_path)
    retriever.fit(units)

    assert retriever.embeddings is not None
    assert retriever.embeddings.shape == (2, DEFAULT_EMBEDDING_DIM)
    assert retriever.unit_ids == ["unit_alpha", "unit_beta"]

    # Verify L2 normalization: norm of each vector is ~1.0
    for vec in retriever.embeddings:
        norm = np.linalg.norm(vec)
        assert pytest.approx(norm, 0.001) == 1.0


def test_embedding_cache_creation_and_hit(tmp_path: Path):
    """Verify embedding cache file is written and successfully re-loaded on subsequent runs."""
    units = [
        {
            "unit_id": "u_test",
            "rel_path": "test.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "testFunc",
            "source_code": "function testFunc() {}",
            "content_sha256": "test_sha_123",
        }
    ]

    retriever1 = SemanticRetriever(cache_dir=tmp_path)
    retriever1.fit(units)
    assert retriever1.embeddings is not None

    # Check cache file exists in tmp_path
    cache_files = list(tmp_path.glob("embeddings_*.npz"))
    assert len(cache_files) == 1

    # Second retriever with same cache_dir should load from disk
    retriever2 = SemanticRetriever(cache_dir=tmp_path)
    # Monkey-patch _get_model to fail if called, verifying it hit disk cache
    retriever2._get_model = lambda: pytest.fail("_get_model was called despite valid disk cache!")
    retriever2.fit(units)

    assert retriever2.embeddings is not None
    assert np.allclose(retriever1.embeddings, retriever2.embeddings)


def test_cache_invalidation_on_content_change(tmp_path: Path):
    """Changing code unit content SHA triggers cache invalidation."""
    units1 = [
        {
            "unit_id": "u1",
            "rel_path": "a.js",
            "source_code": "function a() {}",
            "content_sha256": "initial_sha",
        }
    ]
    retriever = SemanticRetriever(cache_dir=tmp_path)
    retriever.fit(units1)
    cache_file_1 = list(tmp_path.glob("embeddings_*.npz"))[0]

    # Mutate content_sha256
    units2 = [
        {
            "unit_id": "u1",
            "rel_path": "a.js",
            "source_code": "function aModified() {}",
            "content_sha256": "modified_sha",
        }
    ]
    retriever2 = SemanticRetriever(cache_dir=tmp_path)
    retriever2.fit(units2)
    cache_files = list(tmp_path.glob("embeddings_*.npz"))

    # Two separate cache files exist for different fingerprints
    assert len(cache_files) == 2
    assert cache_file_1 in cache_files


def test_deterministic_ranking(tmp_path: Path):
    """Ranking is deterministic across repeated searches."""
    units = [
        {
            "unit_id": "unit_pdf_merge",
            "rel_path": "merge.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "mergePdfs",
            "source_code": "function mergePdfs() { return 'merged'; }",
            "content_sha256": "sha_m",
        },
        {
            "unit_id": "unit_pdf_split",
            "rel_path": "split.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "splitPdf",
            "source_code": "function splitPdf() { return 'split'; }",
            "content_sha256": "sha_s",
        },
    ]

    retriever = SemanticRetriever(cache_dir=tmp_path)
    retriever.fit(units)

    results1 = retriever.search("merge PDF documents", top_k=2)
    results2 = retriever.search("merge PDF documents", top_k=2)

    assert len(results1) == 2
    assert [r.unit_id for r in results1] == [r.unit_id for r in results2]
    assert [r.score for r in results1] == [r.score for r in results2]
    # Semantic match for merge should put unit_pdf_merge at rank 1
    assert results1[0].unit_id == "unit_pdf_merge"


def test_model_loading_error(tmp_path: Path):
    """Invalid or nonexistent model name raises RuntimeError."""
    retriever = SemanticRetriever(
        model_name="nonexistent_invalid_org/model_does_not_exist_xyz123",
        cache_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="Failed to load sentence-embedding model"):
        retriever.fit([
            {
                "unit_id": "u1",
                "rel_path": "a.js",
                "source_code": "code",
                "content_sha256": "sha",
            }
        ])


def test_metric_compatibility(tmp_path: Path):
    """Verify SemanticRetriever is completely compatible with evaluate_dataset harness."""
    units = [
        {
            "unit_id": "u_pdf_merge",
            "rel_path": "pdf.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "mergePdfs",
            "source_code": "function mergePdfs() { /* combine multiple pdfs */ }",
            "content_sha256": "sha_merge",
        }
    ]
    retriever = SemanticRetriever(cache_dir=tmp_path)
    retriever.fit(units)

    queries = [
        EvaluationQuery(
            query_id="q1",
            query="merge multiple PDF files",
            relevant_unit_ids=["u_pdf_merge"],
            task_category="pdf_processing",
        )
    ]

    query_results, aggregate = evaluate_dataset(retriever, queries, top_k=5)
    assert aggregate.evaluated_queries == 1
    assert aggregate.mean_recall_at_1 == 1.0
    assert aggregate.mrr == 1.0
    assert len(query_results) == 1
    assert query_results[0].recall_at_1 == 1.0
