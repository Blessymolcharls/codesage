"""Code-specific Transformer retrieval using pretrained CodeSearchNet models and cosine similarity."""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from codesage.retrieval import SearchResult, load_code_units_from_jsonl, make_preview

# Default configuration for the code-specific Transformer baseline
DEFAULT_CODE_MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
DEFAULT_CODE_EMBEDDING_DIM = 768
DEFAULT_CODE_MAX_SEQ_LENGTH = 512
DEFAULT_CODE_POOLING = "mean"
DEFAULT_CODE_TRUNCATION_POLICY = "truncate_to_max_seq_length_512"


@dataclass(frozen=True)
class CodeModelConfig:
    """Configuration and metadata for the code-specific embedding model."""
    model_name: str = DEFAULT_CODE_MODEL_NAME
    embedding_dim: int = DEFAULT_CODE_EMBEDDING_DIM
    max_seq_length: int = DEFAULT_CODE_MAX_SEQ_LENGTH
    pooling: str = DEFAULT_CODE_POOLING
    truncation_policy: str = DEFAULT_CODE_TRUNCATION_POLICY
    normalize_embeddings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_code_search_document(
    symbol_name: Optional[str],
    rel_path: str,
    source_code: str,
    language: Optional[str] = None,
    unit_type: Optional[str] = None,
    max_chars: int = 2000,
    include_metadata: bool = True,
) -> str:
    """Compose a code document representation suited for code-specific bi-encoder models.
    
    Includes programming comment headers for language, unit type, and file location,
    followed by the source code body bounded to respect the 512 token context window.
    """
    # Bounded source code: truncate to max_chars to comfortably fit within 512 tokens
    code_body = source_code.strip()
    if len(code_body) > max_chars:
        code_body = code_body[:max_chars].rstrip()

    if not include_metadata:
        return code_body

    comment_prefix = "//" if language in ["javascript", "dart", "java", "cpp", "c", "go"] else "#"

    header_parts: List[str] = []
    if language:
        header_parts.append(f"[{language}]")
    if unit_type:
        header_parts.append(unit_type)
    if symbol_name:
        header_parts.append(symbol_name)
    header = f"{comment_prefix} " + " ".join(header_parts) if header_parts else ""

    path_line = f"{comment_prefix} File: {rel_path}" if rel_path else ""

    sections = [s for s in [header, path_line, code_body] if s]
    return "\n".join(sections)


def compute_code_units_fingerprint(
    units: List[Dict[str, Any]],
    config: CodeModelConfig,
    include_metadata: bool = True,
) -> str:
    """Compute a deterministic hash fingerprint representing the exact corpus and model configuration."""
    hasher = hashlib.sha256()
    hasher.update(config.model_name.encode("utf-8"))
    hasher.update(str(config.max_seq_length).encode("utf-8"))
    hasher.update(str(config.normalize_embeddings).encode("utf-8"))
    hasher.update(f"meta:{include_metadata}".encode("utf-8"))

    for u in units:
        uid = u.get("unit_id", "")
        sha = u.get("content_sha256", "")
        hasher.update(uid.encode("utf-8"))
        hasher.update(sha.encode("utf-8"))

    return hasher.hexdigest()


class CodeTransformerRetriever:
    """Code retrieval engine using code-specific pretrained Transformer embeddings (CodeSearchNet)."""

    def __init__(
        self,
        model_name: str = DEFAULT_CODE_MODEL_NAME,
        cache_dir: Optional[Path | str] = None,
        force_recompute: bool = False,
        device: Optional[str] = None,
        include_metadata: bool = True,
    ) -> None:
        self.config = CodeModelConfig(model_name=model_name)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.force_recompute = force_recompute
        self.device = device
        self.include_metadata = include_metadata

        self.units: List[Dict[str, Any]] = []
        self.unit_ids: List[str] = []
        self.unit_map: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, 768)
        self._model: Optional[Any] = None

    def __len__(self) -> int:
        return len(self.units)

    def _get_model(self) -> Any:
        """Lazily load SentenceTransformer model for code, handling errors gracefully."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. Please install it with `pip install sentence-transformers`."
            ) from e

        try:
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.device,
            )
            # Ensure model max sequence length matches config (512 for distilroberta)
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self.config.max_seq_length
            return self._model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load code-specific Transformer model '{self.config.model_name}': {e}"
            ) from e

    def _get_cache_path(self, fingerprint: str) -> Path:
        """Generate a reproducible cache path for the corpus and model fingerprint."""
        slug = self.config.model_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"embeddings_{slug}_{fingerprint[:16]}.npz"

    def _try_load_cache(self, fingerprint: str, expected_uids: List[str]) -> Optional[np.ndarray]:
        """Attempt to load embeddings from disk cache if fingerprint and unit IDs match exactly."""
        if self.force_recompute:
            return None

        cache_path = self._get_cache_path(fingerprint)
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as data:
                cached_fingerprint = str(data["fingerprint"])
                if cached_fingerprint != fingerprint:
                    return None

                cached_uids = [str(x) for x in data["unit_ids"]]
                if cached_uids != expected_uids:
                    return None

                cached_embeddings = data["embeddings"]
                if cached_embeddings.shape != (len(expected_uids), self.config.embedding_dim):
                    return None

                return cached_embeddings.astype(np.float32)
        except Exception:
            # Corrupted cache file: return None to trigger recomputation
            return None

    def _save_cache(self, fingerprint: str, uids: List[str], embeddings: np.ndarray) -> None:
        """Persist computed embeddings to disk cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._get_cache_path(fingerprint)
            np.savez_compressed(
                cache_path,
                fingerprint=fingerprint,
                unit_ids=np.array(uids, dtype=str),
                embeddings=embeddings.astype(np.float32),
                metadata=json.dumps(self.config.to_dict()),
            )
        except Exception:
            # Non-fatal: caching failure should not halt retrieval
            pass

    def fit(self, units: List[Dict[str, Any]]) -> "CodeTransformerRetriever":
        """Index code units and compute or load code embeddings."""
        self.units = units
        self.unit_ids = [u["unit_id"] for u in units]
        self.unit_map = {u["unit_id"]: u for u in units}

        if not units:
            self.embeddings = None
            return self

        fingerprint = compute_code_units_fingerprint(units, self.config, include_metadata=self.include_metadata)

        # Check local cache
        cached_embeddings = self._try_load_cache(fingerprint, self.unit_ids)
        if cached_embeddings is not None:
            self.embeddings = cached_embeddings
            return self

        # Cache miss: generate document strings and encode with model
        documents = [
            prepare_code_search_document(
                symbol_name=u.get("symbol_name"),
                rel_path=u.get("rel_path") or u.get("repository_relative_path", ""),
                source_code=u.get("source_code") or u.get("source_text", ""),
                language=u.get("language"),
                unit_type=u.get("unit_type"),
                include_metadata=self.include_metadata,
            )
            for u in units
        ]

        model = self._get_model()
        raw_embeddings = model.encode(
            documents,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        self.embeddings = raw_embeddings.astype(np.float32)
        self._save_cache(fingerprint, self.unit_ids, self.embeddings)
        return self

    def fit_from_jsonl(self, jsonl_path: Path | str) -> "CodeTransformerRetriever":
        """Load code units from JSONL and compute or load embeddings."""
        units = load_code_units_from_jsonl(jsonl_path)
        return self.fit(units)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search indexed code units using query embedding cosine similarity."""
        query_str = query.strip()
        if not query_str or not self.units or self.embeddings is None:
            return []

        model = self._get_model()
        query_vec = model.encode(
            [query_str],
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        # Cosine similarity via dot product (both unit and query embeddings are L2 normalized)
        scores = np.dot(self.embeddings, query_vec)

        # Deterministic sorting: primary = -score (descending), secondary = unit_id (ascending)
        scored_units: List[Tuple[float, str, int]] = []
        for idx in range(len(self.unit_ids)):
            score = float(scores[idx])
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
