"""Automated tests for CodeSage code-specific Transformer retrieval (CodeSearchNet)."""

import json
from pathlib import Path
import pytest
import numpy as np

from codesage.evaluation import EvaluationQuery, evaluate_dataset
from codesage.code_semantic import (
    DEFAULT_CODE_MODEL_NAME,
    DEFAULT_CODE_EMBEDDING_DIM,
    CodeTransformerRetriever,
    CodeModelConfig,
    compute_code_units_fingerprint,
    prepare_code_search_document,
)


def test_prepare_code_search_document():
    """Verify document preparation formats comment header, path, and bounds source code."""
    doc = prepare_code_search_document(
        symbol_name="mergePdfs",
        rel_path="backend/controllers/pdfController.js",
        source_code="const mergePdfs = async (req, res) => { /* merge logic */ };",
        language="javascript",
        unit_type="function",
        max_chars=60,
    )
    assert "// [javascript] function mergePdfs" in doc
    assert "// File: backend/controllers/pdfController.js" in doc
    assert "const mergePdfs" in doc
    # Verify bounded length
    code_part = doc.split("// File: backend/controllers/pdfController.js\n")[1]
    assert len(code_part) <= 60


def test_empty_inputs_code_transformer(tmp_path: Path):
    """Empty queries, whitespace queries, or empty indexes safely return empty results."""
    retriever = CodeTransformerRetriever(cache_dir=tmp_path)
    assert retriever.search("query") == []

    # Fit with single unit
    retriever.fit([
        {
            "unit_id": "u1",
            "rel_path": "a.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "foo",
            "source_code": "function foo() {}",
            "content_sha256": "abc1",
        }
    ])

    assert retriever.search("") == []
    assert retriever.search("    ") == []
    assert retriever.search("\t\n") == []


def test_stable_mappings_and_vector_dimensions_code_transformer(tmp_path: Path):
    """Verify vectors have 768 dimensions and stably map to unit IDs."""
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

    retriever = CodeTransformerRetriever(cache_dir=tmp_path)
    retriever.fit(units)

    assert retriever.embeddings is not None
    assert retriever.embeddings.shape == (2, DEFAULT_CODE_EMBEDDING_DIM)
    assert retriever.unit_ids == ["unit_alpha", "unit_beta"]

    # Verify L2 normalization
    for vec in retriever.embeddings:
        norm = np.linalg.norm(vec)
        assert pytest.approx(norm, 0.001) == 1.0


def test_code_transformer_embedding_cache_creation_and_hit(tmp_path: Path):
    """Verify embedding cache file is written and successfully re-loaded on subsequent runs."""
    units = [
        {
            "unit_id": "u_code_test",
            "rel_path": "test.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "testFunc",
            "source_code": "function testFunc() {}",
            "content_sha256": "code_test_sha_123",
        }
    ]

    retriever1 = CodeTransformerRetriever(cache_dir=tmp_path)
    retriever1.fit(units)
    assert retriever1.embeddings is not None

    # Check cache file exists in tmp_path
    cache_files = list(tmp_path.glob("embeddings_*.npz"))
    assert len(cache_files) == 1

    # Second retriever with same cache_dir should load from disk
    retriever2 = CodeTransformerRetriever(cache_dir=tmp_path)
    retriever2._get_model = lambda: pytest.fail("_get_model was called despite valid disk cache!")
    retriever2.fit(units)

    assert retriever2.embeddings is not None
    assert np.allclose(retriever1.embeddings, retriever2.embeddings)


def test_code_transformer_cache_invalidation_on_content_change(tmp_path: Path):
    """Changing code unit content SHA triggers cache invalidation."""
    units1 = [
        {
            "unit_id": "u1",
            "rel_path": "a.js",
            "source_code": "function a() {}",
            "content_sha256": "initial_sha",
        }
    ]
    retriever = CodeTransformerRetriever(cache_dir=tmp_path)
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
    retriever2 = CodeTransformerRetriever(cache_dir=tmp_path)
    retriever2.fit(units2)
    cache_files = list(tmp_path.glob("embeddings_*.npz"))

    # Two separate cache files exist for different fingerprints
    assert len(cache_files) == 2
    assert cache_file_1 in cache_files


def test_code_transformer_long_inputs_handling(tmp_path: Path):
    """Verify units with long code (> 512 tokens) are truncated gracefully without error."""
    long_code = "\n".join([f"const var_{i} = calculateSomethingComplex({i});" for i in range(1, 500)])
    units = [
        {
            "unit_id": "u_long",
            "rel_path": "long.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "longFunction",
            "source_code": long_code,
            "content_sha256": "long_sha",
        }
    ]

    retriever = CodeTransformerRetriever(cache_dir=tmp_path)
    retriever.fit(units)

    assert retriever.embeddings is not None
    assert retriever.embeddings.shape == (1, DEFAULT_CODE_EMBEDDING_DIM)

    # Search against long query as well
    results = retriever.search("calculate complex " * 50, top_k=1)
    assert len(results) == 1
    assert results[0].unit_id == "u_long"


def test_code_transformer_deterministic_ranking(tmp_path: Path):
    """Ranking is deterministic across repeated searches."""
    units = [
        {
            "unit_id": "unit_pdf_merge",
            "rel_path": "merge.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "mergePdfs",
            "source_code": "function mergePdfs() { return 'merged'; }",
            "content_sha256": "sha_m",
        },
        {
            "unit_id": "unit_pdf_split",
            "rel_path": "split.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "splitPdf",
            "source_code": "function splitPdf() { return 'split'; }",
            "content_sha256": "sha_s",
        },
    ]

    retriever = CodeTransformerRetriever(cache_dir=tmp_path)
    retriever.fit(units)

    results1 = retriever.search("merge PDF files", top_k=2)
    results2 = retriever.search("merge PDF files", top_k=2)

    assert len(results1) == 2
    assert [r.unit_id for r in results1] == [r.unit_id for r in results2]
    assert [r.score for r in results1] == [r.score for r in results2]


def test_code_transformer_model_loading_error(tmp_path: Path):
    """Invalid or nonexistent model name raises RuntimeError."""
    retriever = CodeTransformerRetriever(
        model_name="nonexistent_invalid_org/code_model_does_not_exist_xyz123",
        cache_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="Failed to load code-specific Transformer model"):
        retriever.fit([
            {
                "unit_id": "u1",
                "rel_path": "a.js",
                "source_code": "code",
                "content_sha256": "sha",
            }
        ])


def test_code_transformer_metric_compatibility(tmp_path: Path):
    """Verify CodeTransformerRetriever is completely compatible with evaluate_dataset harness."""
    units = [
        {
            "unit_id": "u_pdf_merge",
            "rel_path": "pdf.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "mergePdfs",
            "source_code": "function mergePdfs() { /* combine multiple pdfs */ }",
            "content_sha256": "sha_merge",
        }
    ]
    retriever = CodeTransformerRetriever(cache_dir=tmp_path)
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
