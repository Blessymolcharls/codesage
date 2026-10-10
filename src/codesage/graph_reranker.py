"""Graph-context reranking module for CodeSage.

Enhances baseline retrieval results (TF-IDF, general semantic, code Transformer)
by computing graph neighborhood affinity across candidate code units.
Units that have strong structural relationships (class membership, reliable calls,
imported symbols, co-file definitions) with other high-scoring candidates receive
a contextual affinity boost.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from codesage.graph import RelationshipType, RepositoryGraph
from codesage.retrieval import SearchResult


# Default relationship edge affinity weights
DEFAULT_RELATIONSHIP_WEIGHTS: Dict[str, float] = {
    RelationshipType.MEMBER_OF.value: 0.35,
    RelationshipType.CONTAINS_METHOD.value: 0.35,
    RelationshipType.CALLS.value: 0.30,
    RelationshipType.CONSTRUCTS.value: 0.30,
    RelationshipType.EXTENDS.value: 0.25,
    RelationshipType.IMPORTS_SYMBOL.value: 0.25,
    RelationshipType.IMPORTS_FILE.value: 0.15,
    RelationshipType.DEFINES.value: 0.15,
}


class GraphContextReranker:
    """Reranks candidate retrieval results using repository graph structure.
    
    Implements a drop-in retrieval interface (`search(query, top_k)`) wrapping
    any base retriever.
    """

    def __init__(
        self,
        base_retriever: Any,
        graph: RepositoryGraph,
        alpha: float = 0.75,
        beta: float = 0.25,
        candidate_pool_k: int = 25,
        relationship_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize graph reranker.
        
        Args:
            base_retriever: Any retriever implementing `search(query, top_k)`.
            graph: The RepositoryGraph representing code units and relationships.
            alpha: Weight for base retrieval score (default: 0.75).
            beta: Weight for graph neighborhood affinity score (default: 0.25).
            candidate_pool_k: Initial candidate pool depth fetched from base retriever.
            relationship_weights: Custom weights per relationship type.
        """
        self.base_retriever = base_retriever
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.candidate_pool_k = candidate_pool_k
        self.relationship_weights = relationship_weights or DEFAULT_RELATIONSHIP_WEIGHTS

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Retrieve initial candidates using base retriever and rerank using graph context.
        
        Deterministic: ties in reranked score are broken lexicographically by `unit_id`.
        """
        pool_size = max(top_k, self.candidate_pool_k)
        initial_results = self.base_retriever.search(query, top_k=pool_size)

        if not initial_results:
            return []

        if len(initial_results) == 1 or self.beta <= 0.0:
            return initial_results[:top_k]

        return self.rerank(initial_results, top_k=top_k)

    def rerank(self, candidates: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Apply graph affinity reranking on a given list of candidate SearchResults."""
        if not candidates:
            return []

        # Map candidate unit IDs to their initial score and result object
        candidate_units: Dict[str, SearchResult] = {c.unit_id: c for c in candidates}
        base_scores: Dict[str, float] = {c.unit_id: max(0.0, float(c.score)) for c in candidates}

        # Normalize base scores to [0, 1] range for stable combination
        max_base = max(base_scores.values()) if base_scores else 1.0
        min_base = min(base_scores.values()) if base_scores else 0.0
        base_range = max_base - min_base if max_base > min_base else 1.0

        normalized_base = {
            uid: (score - min_base) / base_range for uid, score in base_scores.items()
        }

        # Compute graph neighborhood affinity for each candidate
        raw_graph_scores: Dict[str, float] = {}

        for uid in candidate_units:
            node_id = f"unit:{uid}"
            neighbors = self.graph.get_neighbors(node_id, direction="both")
            affinity = 0.0

            for neighbor_id, rel_type, confidence in neighbors:
                weight = self.relationship_weights.get(rel_type, 0.1) * confidence

                # Direct code-unit neighbor in candidate set
                if neighbor_id.startswith("unit:"):
                    target_uid = neighbor_id[5:]
                    if target_uid in normalized_base and target_uid != uid:
                        affinity += weight * normalized_base[target_uid]

                # File node neighbor: check file's defined units in candidate set
                elif neighbor_id.startswith("file:"):
                    file_neighbors = self.graph.get_neighbors(neighbor_id, direction="outgoing")
                    for fn_id, fn_rel, fn_conf in file_neighbors:
                        if fn_id.startswith("unit:"):
                            f_uid = fn_id[5:]
                            if f_uid in normalized_base and f_uid != uid:
                                # Co-file membership affinity
                                affinity += (weight * 0.5) * normalized_base[f_uid]

            raw_graph_scores[uid] = affinity

        # Normalize graph scores to [0, 1]
        max_graph = max(raw_graph_scores.values()) if raw_graph_scores else 0.0
        norm_graph = {
            uid: (score / max_graph) if max_graph > 0.0 else 0.0
            for uid, score in raw_graph_scores.items()
        }

        # Compute final composite score: alpha * normalized_base + beta * norm_graph
        scored_candidates: List[Tuple[float, str, SearchResult]] = []
        for uid, orig_res in candidate_units.items():
            composite_score = (self.alpha * normalized_base[uid]) + (self.beta * norm_graph[uid])
            # Scale back to realistic score magnitude preserving base scale
            final_score = round(float(orig_res.score) * (1.0 - self.beta) + (norm_graph[uid] * self.beta * max_base), 6)
            scored_candidates.append((composite_score, final_score, orig_res))

        # Deterministic sorting: sort descending by composite_score, then descending by final_score,
        # then ascending by unit_id to break ties reliably
        scored_candidates.sort(key=lambda item: (-item[0], -item[1], item[2].unit_id))

        reranked_results: List[SearchResult] = []
        for comp_score, final_score, orig in scored_candidates[:top_k]:
            reranked_res = SearchResult(
                unit_id=orig.unit_id,
                score=final_score,
                rel_path=orig.rel_path,
                language=orig.language,
                symbol_name=orig.symbol_name,
                unit_type=orig.unit_type,
                start_line=orig.start_line,
                end_line=orig.end_line,
                extraction_method=orig.extraction_method,
                preview=orig.preview,
                source_code=orig.source_code,
            )
            reranked_results.append(reranked_res)

        return reranked_results
