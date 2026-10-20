"""Unit tests for Reciprocal Rank Fusion (RRF) hybrid retrieval and graph integration."""

import pytest

from codesage.graph import RelationshipType, RepositoryGraph
from codesage.hybrid import HybridRetriever, HybridSearchResult, reciprocal_rank_fusion
from codesage.retrieval import SearchResult


def _make_dummy_result(unit_id: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        unit_id=unit_id,
        score=score,
        rel_path=f"path/to/{unit_id}.py",
        language="python",
        symbol_name=unit_id.split(":")[-1] if ":" in unit_id else unit_id,
        unit_type="function",
        start_line=1,
        end_line=10,
        extraction_method="ast",
        preview=f"def {unit_id}(): pass",
        source_code=f"def {unit_id}():\n    pass\n",
    )


class DummyRetriever:
    """Mock retriever returning a predefined list of SearchResults."""
    def __init__(self, results_map):
        self.results_map = results_map
        self.unit_ids = ["unit_a", "unit_b", "unit_c", "unit_d"]

    def search(self, query: str, top_k: int = 10):
        return self.results_map.get(query, [])[:top_k]


def test_rrf_formula_and_1_based_indexing():
    """Verify standard RRF formula RRF(d) = sum(1 / (k + rank_m(d))) with 1-based ranks."""
    r_a1 = _make_dummy_result("unit_a", score=0.9)
    r_b1 = _make_dummy_result("unit_b", score=0.8)

    r_a2 = _make_dummy_result("unit_b", score=0.95)
    r_b2 = _make_dummy_result("unit_a", score=0.7)

    # In model_1: unit_a is rank 1, unit_b is rank 2
    # In model_2: unit_b is rank 1, unit_a is rank 2
    model_rankings = {
        "model_1": [r_a1, r_b1],
        "model_2": [r_a2, r_b2],
    }

    results = reciprocal_rank_fusion(model_rankings, k=60)

    # Both units have 1st and 2nd rank:
    # unit_a: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.032522
    # unit_b: 1/(60+2) + 1/(60+1) = 0.032522
    assert len(results) == 2
    expected_score = round(1.0 / 61.0 + 1.0 / 62.0, 6)
    assert results[0].rrf_score == expected_score
    assert results[1].rrf_score == expected_score

    # Check 1-based model ranks recorded in provenance
    for r in results:
        assert r.model_ranks["model_1"] in [1, 2]
        assert r.model_ranks["model_2"] in [1, 2]
        assert "model_1" in r.rrf_contributions
        assert "model_2" in r.rrf_contributions


def test_rrf_missing_candidates():
    """Candidates absent from a model's top-K do not receive invented ranks or penalties."""
    # Model 1 finds unit_only_1 (rank 1) and unit_both (rank 2)
    # Model 2 finds unit_only_2 (rank 1) and unit_both (rank 2)
    r_only_1 = _make_dummy_result("unit_only_1", score=0.9)
    r_both_1 = _make_dummy_result("unit_both", score=0.8)

    r_only_2 = _make_dummy_result("unit_only_2", score=0.9)
    r_both_2 = _make_dummy_result("unit_both", score=0.8)

    model_rankings = {
        "tfidf": [r_only_1, r_both_1],
        "code": [r_only_2, r_both_2],
    }

    results = reciprocal_rank_fusion(model_rankings, k=60)

    assert len(results) == 3
    # unit_both was retrieved by both models at rank 2:
    # RRF(unit_both) = 1/(60+2) + 1/(60+2) = 2/62 = 0.032258
    # RRF(unit_only_1) = 1/(60+1) = 1/61 = 0.016393
    # RRF(unit_only_2) = 1/(60+1) = 1/61 = 0.016393
    # unit_both should rank first because of consensus across engines!
    assert results[0].unit_id == "unit_both"
    assert results[0].model_ranks == {"tfidf": 2, "code": 2}

    # unit_only_1 should only have tfidf in its model_ranks
    item_only_1 = next(r for r in results if r.unit_id == "unit_only_1")
    assert "tfidf" in item_only_1.model_ranks
    assert "code" not in item_only_1.model_ranks
    assert item_only_1.model_ranks["tfidf"] == 1


def test_rrf_duplicate_candidates_in_single_model():
    """Deduplicate within single model output, only retaining highest rank."""
    r1 = _make_dummy_result("unit_dup", score=0.9)
    r2 = _make_dummy_result("unit_other", score=0.8)
    r3 = _make_dummy_result("unit_dup", score=0.5)  # duplicate of unit_dup

    model_rankings = {
        "model_a": [r1, r2, r3],
    }

    results = reciprocal_rank_fusion(model_rankings, k=60)
    assert len(results) == 2
    u_dup = next(r for r in results if r.unit_id == "unit_dup")
    assert u_dup.model_ranks["model_a"] == 1
    # Score should be exactly 1/(60+1), NOT 1/(60+1) + 1/(60+3)
    assert u_dup.rrf_score == round(1.0 / 61.0, 6)


def test_rrf_deterministic_ordering_and_tie_breaking():
    """Ties in RRF score must be broken deterministically by consensus count, then unit_id."""
    # Both units have identical RRF score (rank 1 in one model each)
    r_z = _make_dummy_result("unit_z", score=0.9)
    r_a = _make_dummy_result("unit_a", score=0.9)

    model_rankings = {
        "model_1": [r_z],
        "model_2": [r_a],
    }

    # RRF for both is 1/61. unit_a should come before unit_z alphabetically.
    results = reciprocal_rank_fusion(model_rankings, k=60)
    assert [r.unit_id for r in results] == ["unit_a", "unit_z"]

    # Shuffled input should produce the exact same order
    model_rankings_shuffled = {
        "model_2": [r_a],
        "model_1": [r_z],
    }
    results_shuffled = reciprocal_rank_fusion(model_rankings_shuffled, k=60)
    assert [r.unit_id for r in results_shuffled] == ["unit_a", "unit_z"]


def test_rrf_empty_rankings():
    """Empty inputs should return empty results safely without exceptions."""
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"m1": [], "m2": []}) == []

    # One empty, one non-empty
    r1 = _make_dummy_result("unit_1", score=0.9)
    results = reciprocal_rank_fusion({"m1": [r1], "m2": []}, k=60)
    assert len(results) == 1
    assert results[0].unit_id == "unit_1"


def test_rrf_configurable_fusion_constant():
    """Verify configurable fusion constant k scales properly."""
    r1 = _make_dummy_result("unit_1", score=0.9)
    r2 = _make_dummy_result("unit_2", score=0.8)

    rankings = {"model": [r1, r2]}

    # Test with k=10
    results_10 = reciprocal_rank_fusion(rankings, k=10)
    assert results_10[0].rrf_score == round(1.0 / 11.0, 6)
    assert results_10[1].rrf_score == round(1.0 / 12.0, 6)

    # Test with k=1
    results_1 = reciprocal_rank_fusion(rankings, k=1)
    assert results_1[0].rrf_score == round(1.0 / 2.0, 6)
    assert results_1[1].rrf_score == round(1.0 / 3.0, 6)

    # Invalid k <= 0 must raise ValueError
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings, k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings, k=-5)


def test_hybrid_retriever_end_to_end_search():
    """Test HybridRetriever combining two mock retrievers."""
    retriever_1 = DummyRetriever({
        "query": [_make_dummy_result("unit_a", 0.9), _make_dummy_result("unit_b", 0.7)],
    })
    retriever_2 = DummyRetriever({
        "query": [_make_dummy_result("unit_b", 0.95), _make_dummy_result("unit_c", 0.6)],
    })

    hybrid = HybridRetriever(
        retrievers={"tfidf": retriever_1, "code": retriever_2},
        k=60,
    )

    results = hybrid.search("query", top_k=5)
    assert len(results) == 3

    # unit_b was rank 2 in tfidf, rank 1 in code:
    # RRF(unit_b) = 1/62 + 1/61 = 0.032522
    # unit_a was rank 1 in tfidf: 1/61 = 0.016393
    # unit_c was rank 2 in code: 1/62 = 0.016129
    assert results[0].unit_id == "unit_b"
    assert isinstance(results[0], HybridSearchResult)
    assert results[0].model_ranks == {"tfidf": 2, "code": 1}
    assert results[1].unit_id == "unit_a"
    assert results[2].unit_id == "unit_c"


def test_hybrid_retriever_optional_graph_reranking():
    """Test optional graph context reranking on top of fused results preserves provenance."""
    retriever_1 = DummyRetriever({
        "query": [_make_dummy_result("unit_a", 0.9), _make_dummy_result("unit_b", 0.8)],
    })
    retriever_2 = DummyRetriever({
        "query": [_make_dummy_result("unit_a", 0.85), _make_dummy_result("unit_c", 0.7)],
    })

    from codesage.graph import GraphEdge, GraphNode

    # Build small graph where unit_b is connected to unit_a via CALLS
    graph = RepositoryGraph()
    for uid in ["unit_a", "unit_b", "unit_c"]:
        graph.add_node(
            GraphNode(
                id=f"unit:{uid}",
                node_type="code_unit",
                name=uid,
                path=f"path/to/{uid}.py",
                language="python",
                unit_type="function",
            )
        )
    graph.add_edge(
        GraphEdge(
            source="unit:unit_b",
            target="unit:unit_a",
            relationship=RelationshipType.CALLS.value,
            confidence=1.0,
        )
    )

    hybrid = HybridRetriever(
        retrievers={"tfidf": retriever_1, "code": retriever_2},
        k=60,
        graph=graph,
        rerank_alpha=0.7,
        rerank_beta=0.3,
    )

    # Search with rerank=False
    results_base = hybrid.search("query", top_k=3, rerank=False)
    assert results_base[0].graph_boost == 0.0

    # Search with rerank=True
    results_reranked = hybrid.search("query", top_k=3, rerank=True)
    assert len(results_reranked) == 3
    # All items are HybridSearchResult with provenance
    for r in results_reranked:
        assert isinstance(r, HybridSearchResult)
        assert r.rrf_score > 0.0
        assert hasattr(r, "graph_boost")
        assert hasattr(r, "graph_score")
