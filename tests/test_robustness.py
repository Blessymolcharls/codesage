"""Automated tests for robustness: empty indexes, invalid queries, missing models, and edge cases."""

import pytest
from pathlib import Path
from codesage.retrieval import TfidfRetriever, SearchResult
from codesage.semantic import SemanticRetriever
from codesage.code_semantic import CodeTransformerRetriever
from codesage.graph import RepositoryGraph, RepositoryGraphBuilder, GraphNode, GraphEdge
from codesage.graph_reranker import GraphContextReranker


def test_empty_index_retrievers(tmp_path: Path):
    """Verify all retrievers handle empty unit index ([]) safely without errors."""
    tfidf = TfidfRetriever().fit([])
    assert len(tfidf) == 0
    assert tfidf.search("merge PDF") == []

    sem = SemanticRetriever(cache_dir=tmp_path).fit([])
    assert len(sem) == 0
    assert sem.search("merge PDF") == []

    code = CodeTransformerRetriever(cache_dir=tmp_path).fit([])
    assert len(code) == 0
    assert code.search("merge PDF") == []


def test_special_characters_and_symbols_in_queries(tmp_path: Path):
    """Verify search handles symbols, punctuation, regex characters, and code tokens safely."""
    units = [
        {
            "unit_id": "u1",
            "rel_path": "lib/test.dart",
            "language": "dart",
            "unit_type": "method",
            "symbol_name": "_processFiles",
            "source_code": "Future<void> _processFiles() async { if (x == null) return; }",
            "content_sha256": "hash1",
        }
    ]

    tfidf = TfidfRetriever().fit(units)
    # Regex special characters and punctuation
    queries = [
        "Future<void>",
        "_processFiles()",
        "x == null",
        "*** [test] (regex) +? ^$ ***",
        "// comment with @annotation",
        "🔥 Unicode emoji test",
    ]
    for q in queries:
        res = tfidf.search(q, top_k=5)
        # Should not crash; can return 0 or more results
        assert isinstance(res, list)


def test_graph_reranker_with_empty_or_disconnected_graph():
    """Verify GraphContextReranker handles empty graph or candidates without edges safely."""
    empty_graph = RepositoryGraph()

    class MockBaseRetriever:
        def search(self, query: str, top_k: int = 10):
            return [
                SearchResult(
                    unit_id="isolated_unit_1",
                    score=0.85,
                    rel_path="a.js",
                    language="javascript",
                    symbol_name="a",
                    unit_type="function",
                    start_line=1,
                    end_line=10,
                    extraction_method="tree_sitter",
                    preview="preview",
                    source_code="source",
                )
            ]

    reranker = GraphContextReranker(MockBaseRetriever(), empty_graph)
    results = reranker.search("test", top_k=5)
    assert len(results) == 1
    assert results[0].unit_id == "isolated_unit_1"
    assert results[0].score == 0.85


def test_graph_reranker_empty_results():
    """Verify GraphContextReranker handles base retriever returning no results."""
    empty_graph = RepositoryGraph()

    class EmptyMockRetriever:
        def search(self, query: str, top_k: int = 10):
            return []

    reranker = GraphContextReranker(EmptyMockRetriever(), empty_graph)
    assert reranker.search("anything", top_k=5) == []
    assert reranker.rerank([]) == []
