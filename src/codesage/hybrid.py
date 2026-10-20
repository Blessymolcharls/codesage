"""Hybrid retrieval using Reciprocal Rank Fusion (RRF) and optional graph reranking."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from codesage.graph import RepositoryGraph
from codesage.graph_reranker import GraphContextReranker
from codesage.retrieval import SearchResult


@dataclass(frozen=True)
class HybridSearchResult(SearchResult):
    """Ranked search result produced by Reciprocal Rank Fusion, with inspectable provenance."""
    rrf_score: float = 0.0
    model_ranks: Dict[str, int] = field(default_factory=dict)
    model_scores: Dict[str, float] = field(default_factory=dict)
    rrf_contributions: Dict[str, float] = field(default_factory=dict)
    graph_boost: float = 0.0
    graph_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rrf_score"] = self.rrf_score
        d["model_ranks"] = dict(self.model_ranks)
        d["model_scores"] = dict(self.model_scores)
        d["rrf_contributions"] = dict(self.rrf_contributions)
        d["graph_boost"] = self.graph_boost
        d["graph_score"] = self.graph_score
        return d


def reciprocal_rank_fusion(
    model_rankings: Dict[str, List[SearchResult]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[HybridSearchResult]:
    """Combine ranked outputs from multiple retrieval engines using Reciprocal Rank Fusion.

    Standard formula:
        RRF(d) = sum_{m in M, d in m} (w_m / (k + rank_m(d)))

    Conventions and Guarantees:
    1. Rank indexing is strictly 1-based: rank_m(d) in {1, 2, ... len(results_m)}.
    2. Missing documents: if document d is absent from model m's candidates, no term
       is added. It does NOT receive an invented rank or penalty rank.
    3. Duplicate candidates: if a model produces duplicate unit IDs, only the highest
       rank (first appearance) is counted; later duplicates are discarded.
    4. Deterministic sorting: ties in composite RRF score are broken deterministically
       by consensus count (number of models retrieving the unit), then by unit_id ascending.
    5. Configurable fusion constant k: must be a positive integer (standard default: 60).

    Args:
        model_rankings: Dictionary mapping model names (e.g. 'tfidf', 'code') to ranked SearchResults.
        k: Smoothing constant for reciprocal rank (default: 60, k > 0).
        weights: Optional dictionary of weights w_m per model (defaults to 1.0 for all models).

    Returns:
        List of HybridSearchResult instances ordered deterministically by fused rank.
    """
    if k <= 0:
        raise ValueError(f"Reciprocal Rank Fusion constant k must be positive, got {k}")

    if not model_rankings:
        return []

    model_weights = weights or {}

    # Accumulate unit metadata, ranks, and contributions
    unit_templates: Dict[str, SearchResult] = {}
    unit_ranks: Dict[str, Dict[str, int]] = {}
    unit_scores: Dict[str, Dict[str, float]] = {}
    unit_contributions: Dict[str, Dict[str, float]] = {}
    rrf_totals: Dict[str, float] = {}

    for model_name, results in model_rankings.items():
        if not results:
            continue

        w_m = float(model_weights.get(model_name, 1.0))
        seen_in_model: Set[str] = set()

        for rank_idx, result in enumerate(results, start=1):  # 1-based rank
            uid = result.unit_id
            if uid in seen_in_model:
                continue  # Deduplicate within model output, preserving best rank
            seen_in_model.add(uid)

            if uid not in unit_templates:
                unit_templates[uid] = result
                unit_ranks[uid] = {}
                unit_scores[uid] = {}
                unit_contributions[uid] = {}
                rrf_totals[uid] = 0.0

            contribution = w_m / float(k + rank_idx)
            unit_ranks[uid][model_name] = rank_idx
            unit_scores[uid][model_name] = float(result.score)
            unit_contributions[uid][model_name] = round(contribution, 6)
            rrf_totals[uid] += contribution

    if not unit_templates:
        return []

    # Deterministic ordering:
    # 1. -rrf_score (descending)
    # 2. -len(model_ranks) (descending: consensus across more models ranks higher)
    # 3. unit_id (ascending: alphabetical tie-break)
    sorted_candidates: List[Tuple[float, int, str]] = [
        (-rrf_totals[uid], -len(unit_ranks[uid]), uid)
        for uid in unit_templates
    ]
    sorted_candidates.sort()

    output: List[HybridSearchResult] = []
    for neg_score, neg_consensus, uid in sorted_candidates:
        orig = unit_templates[uid]
        score = round(-neg_score, 6)
        output.append(
            HybridSearchResult(
                unit_id=orig.unit_id,
                score=score,
                rel_path=orig.rel_path,
                language=orig.language,
                symbol_name=orig.symbol_name,
                unit_type=orig.unit_type,
                start_line=orig.start_line,
                end_line=orig.end_line,
                extraction_method=orig.extraction_method,
                preview=orig.preview,
                source_code=orig.source_code,
                rrf_score=score,
                model_ranks=dict(unit_ranks[uid]),
                model_scores=dict(unit_scores[uid]),
                rrf_contributions=dict(unit_contributions[uid]),
                graph_boost=0.0,
                graph_score=0.0,
            )
        )

    return output


class HybridRetriever:
    """Multi-engine hybrid retriever combining lexical and dense retrieval via Reciprocal Rank Fusion.

    Supports optional graph-context neighborhood affinity reranking after fusion.
    """

    def __init__(
        self,
        retrievers: Dict[str, Any],
        k: int = 60,
        candidate_depth: int = 25,
        weights: Optional[Dict[str, float]] = None,
        graph: Optional[RepositoryGraph] = None,
        rerank_alpha: float = 0.75,
        rerank_beta: float = 0.25,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            retrievers: Mapping of model names to retriever instances (e.g. {'tfidf': tfidf, 'code': code_t}).
            k: Reciprocal rank fusion constant (default: 60).
            candidate_depth: Depth of candidates retrieved from each constituent model before fusion.
            weights: Optional model weighting factors.
            graph: Optional RepositoryGraph for structural reranking.
            rerank_alpha: Weight for base RRF score in graph reranking.
            rerank_beta: Weight for graph neighborhood affinity in graph reranking.
        """
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one constituent retriever")
        if k <= 0:
            raise ValueError(f"Fusion constant k must be positive, got {k}")

        self.retrievers = retrievers
        self.k = k
        self.candidate_depth = candidate_depth
        self.weights = weights or {}
        self.graph = graph
        self.rerank_alpha = rerank_alpha
        self.rerank_beta = rerank_beta

        self.reranker: Optional[GraphContextReranker] = None
        if self.graph:
            self.reranker = GraphContextReranker(
                base_retriever=self,
                graph=self.graph,
                alpha=self.rerank_alpha,
                beta=self.rerank_beta,
                candidate_pool_k=candidate_depth,
            )

    @property
    def unit_ids(self) -> List[str]:
        """Expose unit_ids from first available retriever for evaluation harness compatibility."""
        for r in self.retrievers.values():
            if hasattr(r, "unit_ids"):
                return r.unit_ids
        return []

    def search_fused_candidates(self, query: str, pool_size: int) -> List[HybridSearchResult]:
        """Retrieve candidate pools from each model and perform RRF fusion."""
        clean_query = query.strip()
        if not clean_query:
            return []

        model_rankings: Dict[str, List[SearchResult]] = {}
        for name, retriever in self.retrievers.items():
            try:
                results = retriever.search(clean_query, top_k=pool_size)
                model_rankings[name] = results
            except Exception:
                # If a constituent model fails, proceed with remaining retrievers
                model_rankings[name] = []

        return reciprocal_rank_fusion(model_rankings, k=self.k, weights=self.weights)

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
    ) -> List[SearchResult]:
        """Execute hybrid search using RRF fusion and optional graph reranking.

        Args:
            query: Natural language or keyword query string.
            top_k: Number of ranked results to return.
            rerank: Whether to apply graph context reranking to fused results.

        Returns:
            List of SearchResult or HybridSearchResult items.
        """
        pool_size = max(top_k, self.candidate_depth)
        fused = self.search_fused_candidates(query, pool_size=pool_size)

        if not fused:
            return []

        if not rerank or not self.graph or self.rerank_beta <= 0.0 or len(fused) <= 1:
            return fused[:top_k]

        # Apply graph context reranking to the fused candidates
        return self._rerank_fused(fused, top_k=top_k)

    def _rerank_fused(
        self,
        fused_candidates: List[HybridSearchResult],
        top_k: int = 10,
    ) -> List[HybridSearchResult]:
        """Apply graph neighborhood affinity reranking to fused candidates while preserving provenance."""
        if not self.graph:
            return fused_candidates[:top_k]

        candidate_map: Dict[str, HybridSearchResult] = {c.unit_id: c for c in fused_candidates}
        base_scores: Dict[str, float] = {c.unit_id: float(c.score) for c in fused_candidates}

        max_base = max(base_scores.values()) if base_scores else 1.0
        min_base = min(base_scores.values()) if base_scores else 0.0
        base_range = max_base - min_base if max_base > min_base else 1.0

        normalized_base = {
            uid: (score - min_base) / base_range for uid, score in base_scores.items()
        }

        # Compute graph neighborhood affinity
        rel_weights = (
            self.reranker.relationship_weights if self.reranker else None
        ) or {}

        raw_graph_scores: Dict[str, float] = {}
        for uid in candidate_map:
            node_id = f"unit:{uid}"
            neighbors = self.graph.get_neighbors(node_id, direction="both")
            affinity = 0.0

            for neighbor_id, rel_type, confidence in neighbors:
                weight = rel_weights.get(rel_type, 0.1) * confidence

                if neighbor_id.startswith("unit:"):
                    target_uid = neighbor_id[5:]
                    if target_uid in normalized_base and target_uid != uid:
                        affinity += weight * normalized_base[target_uid]

                elif neighbor_id.startswith("file:"):
                    file_neighbors = self.graph.get_neighbors(neighbor_id, direction="outgoing")
                    for fn_id, fn_rel, fn_conf in file_neighbors:
                        if fn_id.startswith("unit:"):
                            f_uid = fn_id[5:]
                            if f_uid in normalized_base and f_uid != uid:
                                affinity += (weight * 0.5) * normalized_base[f_uid]

            raw_graph_scores[uid] = affinity

        max_graph = max(raw_graph_scores.values()) if raw_graph_scores else 0.0
        norm_graph = {
            uid: (score / max_graph) if max_graph > 0.0 else 0.0
            for uid, score in raw_graph_scores.items()
        }

        # Compute composite score
        scored: List[Tuple[float, float, str]] = []
        for uid, orig in candidate_map.items():
            comp_score = (self.rerank_alpha * normalized_base[uid]) + (self.rerank_beta * norm_graph[uid])
            final_score = round(
                float(orig.score) * (1.0 - self.rerank_beta) + (norm_graph[uid] * self.rerank_beta * max_base),
                6,
            )
            scored.append((comp_score, final_score, uid))

        # Deterministic sort: descending comp_score, descending final_score, ascending unit_id
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

        reranked_results: List[HybridSearchResult] = []
        for comp_score, final_score, uid in scored[:top_k]:
            orig = candidate_map[uid]
            boost = round(final_score - orig.rrf_score, 6)
            reranked_results.append(
                HybridSearchResult(
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
                    rrf_score=orig.rrf_score,
                    model_ranks=dict(orig.model_ranks),
                    model_scores=dict(orig.model_scores),
                    rrf_contributions=dict(orig.rrf_contributions),
                    graph_boost=boost,
                    graph_score=round(norm_graph[uid], 4),
                )
            )

        return reranked_results
