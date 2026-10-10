"""Automated unit and integration tests for CodeSage TF-IDF retrieval."""

import json
from pathlib import Path
import pytest

from codesage.retrieval import (
    TfidfRetriever,
    SearchResult,
    make_preview,
    load_code_units_from_jsonl,
)
from codesage.tokenization import code_aware_tokenize, prepare_search_document


def test_code_aware_tokenization():
    """Verify code-aware tokenization splits CamelCase, snake_case, and keeps compound tokens."""
    text = "class PdfMergeService { validateUploadedPdf(file_path, max_size) }"
    tokens = code_aware_tokenize(text)

    # Both compound and decomposed parts must exist
    assert "pdfmergeservice" in tokens
    assert "merge" in tokens
    assert "service" in tokens
    assert "validateuploadedpdf" in tokens
    assert "validate" in tokens
    assert "uploaded" in tokens
    assert "pdf" in tokens
    assert "file_path" in tokens
    assert "file" in tokens
    assert "path" in tokens


def test_prepare_search_document_weights():
    """Verify symbol and path tokens appear multiple times for relevance weighting."""
    doc = prepare_search_document(
        symbol_name="rotatePdfPage",
        rel_path="backend/services/pdfService.js",
        source_code="function rotatePdfPage(angle) { return angle; }",
    )
    tokens = code_aware_tokenize(doc)
    # 'rotate' should appear several times due to symbol repeats and source code
    assert tokens.count("rotate") >= 3
    assert "rotatepdfpage" in tokens
    assert "pdfservice" in tokens


def test_retrieval_synthetic_ranking(tmp_path: Path):
    """Verify that specific queries rank the intended synthetic code units at rank 1."""
    units = [
        {
            "unit_id": "unit_pdf_merge",
            "rel_path": "lib/merge.dart",
            "language": "dart",
            "unit_type": "function",
            "symbol_name": "mergeMultiplePdfs",
            "start_line": 1,
            "end_line": 15,
            "source_code": "Future<File> mergeMultiplePdfs(List<File> pdfFiles) async {\n  final document = PdfDocument();\n  for (final f in pdfFiles) { document.addPages(f); }\n  return document.save();\n}",
            "extraction_method": "tree_sitter",
        },
        {
            "unit_id": "unit_db_connect",
            "rel_path": "server/database.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "connectToDatabase",
            "start_line": 1,
            "end_line": 10,
            "source_code": "async function connectToDatabase(connectionUri) {\n  const client = new MongoClient(connectionUri);\n  await client.connect();\n  return client.db();\n}",
            "extraction_method": "tree_sitter",
        },
        {
            "unit_id": "unit_pdf_validate",
            "rel_path": "server/validator.js",
            "language": "javascript",
            "unit_type": "function",
            "symbol_name": "validateUploadedPdf",
            "start_line": 1,
            "end_line": 12,
            "source_code": "function validateUploadedPdf(file) {\n  if (!file.mimetype.includes('pdf')) throw new Error('Invalid file type');\n  return true;\n}",
            "extraction_method": "tree_sitter",
        },
    ]

    index_file = tmp_path / "synthetic_units.jsonl"
    with open(index_file, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")

    retriever = TfidfRetriever()
    retriever.fit_from_jsonl(index_file)
    assert len(retriever) == 3

    # Query 1: Database connection
    db_results = retriever.search("connect to the database", top_k=3)
    assert len(db_results) > 0
    assert db_results[0].unit_id == "unit_db_connect"
    assert db_results[0].score > 0.0

    # Query 2: Merge PDF
    merge_results = retriever.search("merge multiple PDF files", top_k=3)
    assert len(merge_results) > 0
    assert merge_results[0].unit_id == "unit_pdf_merge"
    assert merge_results[0].score > 0.0

    # Query 3: Validate PDF
    val_results = retriever.search("validate an uploaded PDF", top_k=3)
    assert len(val_results) > 0
    assert val_results[0].unit_id == "unit_pdf_validate"
    assert val_results[0].score > 0.0


def test_empty_query_and_whitespace():
    """Empty or whitespace queries return empty results without error."""
    retriever = TfidfRetriever()
    retriever.fit([
        {
            "unit_id": "u1",
            "rel_path": "a.js",
            "language": "js",
            "unit_type": "file",
            "symbol_name": None,
            "start_line": 1,
            "end_line": 2,
            "source_code": "console.log('hello');",
            "extraction_method": "fallback",
        }
    ])
    assert retriever.search("") == []
    assert retriever.search("    ") == []
    assert retriever.search(" \t \n ") == []


def test_out_of_vocabulary_query():
    """Query with words unseen in the corpus returns empty results or 0 scores gracefully."""
    retriever = TfidfRetriever()
    retriever.fit([
        {
            "unit_id": "u1",
            "rel_path": "a.js",
            "language": "js",
            "unit_type": "file",
            "symbol_name": None,
            "start_line": 1,
            "end_line": 2,
            "source_code": "function calculateSum(a, b) { return a + b; }",
            "extraction_method": "fallback",
        }
    ])
    # Totally out-of-vocabulary words
    results = retriever.search("xylophone quantum astrophysics")
    assert results == []


def test_deterministic_ranking_tie_breaking():
    """Verify ranking is deterministic even when scores are identical."""
    retriever = TfidfRetriever()
    # Two identical code units with different unit_ids
    units = [
        {
            "unit_id": "unit_b",
            "rel_path": "file.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "foo",
            "start_line": 1,
            "end_line": 2,
            "source_code": "function foo() {}",
            "extraction_method": "tree_sitter",
        },
        {
            "unit_id": "unit_a",
            "rel_path": "file.js",
            "language": "js",
            "unit_type": "function",
            "symbol_name": "foo",
            "start_line": 1,
            "end_line": 2,
            "source_code": "function foo() {}",
            "extraction_method": "tree_sitter",
        },
    ]
    retriever.fit(units)

    results1 = retriever.search("foo", top_k=2)
    results2 = retriever.search("foo", top_k=2)

    # Identical score; tie-breaker is unit_id ascending ("unit_a" before "unit_b")
    assert [r.unit_id for r in results1] == ["unit_a", "unit_b"]
    assert [r.unit_id for r in results2] == ["unit_a", "unit_b"]


def test_empty_index_handling(tmp_path: Path):
    """Empty index file or 0 units returns empty search results gracefully."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    retriever = TfidfRetriever()
    retriever.fit_from_jsonl(empty_file)
    assert len(retriever) == 0

    results = retriever.search("test query")
    assert results == []


def test_invalid_jsonl_skipping(tmp_path: Path):
    """Corrupt or non-JSON lines are skipped gracefully with valid lines preserved."""
    corrupt_file = tmp_path / "corrupt.jsonl"
    corrupt_file.write_text(
        "NOT JSON\n"
        '{"unit_id": "valid_1", "rel_path": "valid.js", "language": "js", "unit_type": "function", "symbol_name": "valid", "start_line": 1, "end_line": 2, "source_code": "function valid() {}", "extraction_method": "fallback"}\n'
        "{broken json{\n",
        encoding="utf-8",
    )

    units = load_code_units_from_jsonl(corrupt_file)
    assert len(units) == 1
    assert units[0]["unit_id"] == "valid_1"


def test_preview_bounding():
    """Verify code preview adheres to max_lines and truncates gracefully."""
    long_code = "\n".join([f"line_{i}" for i in range(1, 30)])
    preview = make_preview(long_code, max_lines=6)
    lines = preview.splitlines()
    assert len(lines) == 7  # 6 code lines + '... (23 more lines)'
    assert "more lines" in lines[-1]


def test_char_ngrams_configuration():
    """Verify subword character n-grams can match partial sub-token stems."""
    retriever = TfidfRetriever(use_char_ngrams=True)
    retriever.fit([
        {
            "unit_id": "u_rot",
            "rel_path": "rotate.dart",
            "language": "dart",
            "unit_type": "function",
            "symbol_name": "rotatePdfPage",
            "start_line": 1,
            "end_line": 5,
            "source_code": "void rotatePdfPage() {}",
            "extraction_method": "fallback",
        }
    ])
    results = retriever.search("rotat")
    assert len(results) == 1
    assert results[0].unit_id == "u_rot"
