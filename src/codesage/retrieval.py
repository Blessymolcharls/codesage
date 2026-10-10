"""TF-IDF lexical code-search engine using scikit-learn and cosine similarity."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from codesage.tokenization import code_aware_tokenize, prepare_search_document


@dataclass(frozen=True)
class SearchResult:
    """Individual ranked search result for a code unit."""
    unit_id: str
    score: float
    rel_path: str
    language: str
    symbol_name: Optional[str]
    unit_type: str
    start_line: int
    end_line: int
    extraction_method: str
    preview: str
    source_code: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_preview(source_code: str, max_lines: int = 8) -> str:
    """Produce a bounded, readable source code preview."""
    lines = source_code.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    preview_lines = lines[:max_lines]
    remaining = len(lines) - max_lines
    preview_lines.append(f"... ({remaining} more lines)")
    return "\n".join(preview_lines)


def load_code_units_from_jsonl(jsonl_path: Path | str) -> List[Dict[str, Any]]:
    """Safely load code unit records from a JSONL index file."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")

    units: List[Dict[str, Any]] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                unit_id = data.get("unit_id")
                if not unit_id or unit_id in seen_ids:
                    continue  # Deduplicate records safely
                seen_ids.add(unit_id)
                units.append(data)
            except json.JSONDecodeError:
                # Log or skip malformed line gracefully
                continue

    return units


class TfidfRetriever:
    """Lexical code retrieval engine based on TF-IDF representation and cosine similarity."""

    def __init__(
        self,
        use_char_ngrams: bool = False,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True,
        include_metadata: bool = True,
    ) -> None:
        self.use_char_ngrams = use_char_ngrams
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.include_metadata = include_metadata

        self.units: List[Dict[str, Any]] = []
        self.unit_ids: List[str] = []
        self.unit_map: Dict[str, Dict[str, Any]] = {}
        self.vectorizer: Optional[Any] = None
        self.tfidf_matrix: Optional[Any] = None

    def __len__(self) -> int:
        return len(self.units)

    def fit(self, units: List[Dict[str, Any]]) -> "TfidfRetriever":
        """Fit TF-IDF representations on a collection of code unit dictionaries."""
        self.units = units
        self.unit_ids = [u["unit_id"] for u in units]
        self.unit_map = {u["unit_id"]: u for u in units}

        if not units:
            self.vectorizer = None
            self.tfidf_matrix = None
            return self

        # Prepare document representations
        documents = [
            prepare_search_document(
                symbol_name=u.get("symbol_name"),
                rel_path=u.get("rel_path") or u.get("repository_relative_path", ""),
                source_code=u.get("source_code") or u.get("source_text", ""),
                language=u.get("language"),
                unit_type=u.get("unit_type"),
                include_metadata=self.include_metadata,
            )
            for u in units
        ]

        if self.use_char_ngrams:
            # Union of word-level ngrams and character 3-5 ngrams for subword matching
            word_vec = TfidfVectorizer(
                tokenizer=code_aware_tokenize,
                token_pattern=None,
                lowercase=False,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                sublinear_tf=self.sublinear_tf,
                norm="l2",
            )
            char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=self.min_df,
                sublinear_tf=self.sublinear_tf,
                norm="l2",
            )
            self.vectorizer = FeatureUnion([("word", word_vec), ("char", char_vec)])
        else:
            self.vectorizer = TfidfVectorizer(
                tokenizer=code_aware_tokenize,
                token_pattern=None,
                lowercase=False,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                sublinear_tf=self.sublinear_tf,
                norm="l2",
            )

        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        return self

    def fit_from_jsonl(self, jsonl_path: Path | str) -> "TfidfRetriever":
        """Load code units from JSONL and fit the vectorizer."""
        units = load_code_units_from_jsonl(jsonl_path)
        return self.fit(units)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search indexed code units and return ranked results with deterministic tie-breaking."""
        query_str = query.strip()
        if not query_str or not self.units or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([query_str])

        # Compute cosine similarities (both document and query vectors are L2-normalized)
        # Result shape: (1, n_docs)
        sim_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()

        # Filter strictly positive similarity
        non_zero_indices = np.where(sim_scores > 1e-6)[0]
        if len(non_zero_indices) == 0:
            return []

        # Deterministic sorting: primary = -score (descending), secondary = unit_id (ascending)
        scored_units: List[Tuple[float, str, int]] = []
        for idx in non_zero_indices:
            score = float(sim_scores[idx])
            unit_id = self.unit_ids[idx]
            scored_units.append((-score, unit_id, idx))

        scored_units.sort()

        # Build SearchResult objects
        results: List[SearchResult] = []
        for neg_score, unit_id, idx in scored_units[:top_k]:
            unit = self.units[idx]
            rel_path = unit.get("rel_path") or unit.get("repository_relative_path", "")
            source_code = unit.get("source_code") or unit.get("source_text", "")
            preview = make_preview(source_code)
            results.append(
                SearchResult(
                    unit_id=unit_id,
                    score=float(-neg_score),
                    rel_path=rel_path,
                    language=unit.get("language", ""),
                    symbol_name=unit.get("symbol_name"),
                    unit_type=unit.get("unit_type", ""),
                    start_line=unit.get("start_line", 0),
                    end_line=unit.get("end_line", 0),
                    extraction_method=unit.get("extraction_method", ""),
                    preview=preview,
                    source_code=source_code,
                )
            )

        return results
