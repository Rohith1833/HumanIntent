"""Lightweight local SentenceTransformer semantic layer for HumanIntent.

Core Invariants:
- Fully local and CPU-optimized (runs without GPU or external APIs).
- Model is loaded once and cached in process memory.
- Completely offline after initial local download.
- No user data is ever sent externally.
- Fast execution (single inference < 15ms on modern CPU).
"""

import logging
import threading
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger("humanintent.semantic_matcher")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticMatcher:
    """Reusable local semantic embedding and similarity service."""

    _instance: Optional["SemanticMatcher"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._initialized = False

    def _ensure_model_loaded(self) -> None:
        """Lazy thread-safe loading of the SentenceTransformer model."""
        if self._initialized and self._model is not None:
            return

        with self._lock:
            if self._initialized and self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model '%s' locally...", self.model_name)
                # Load locally; check local cache first for instant offline loading
                try:
                    self._model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(self.model_name, device="cpu")
                self._initialized = True
                logger.info("SentenceTransformer model '%s' loaded successfully.", self.model_name)
            except Exception as e:
                logger.error("Failed to load SentenceTransformer '%s': %s", self.model_name, e)
                self._model = None
                self._initialized = True

    @property
    def is_available(self) -> bool:
        """Check if local embedding model is ready."""
        self._ensure_model_loaded()
        return self._model is not None

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a normalized 1D embedding vector."""
        if not text or not text.strip():
            return np.zeros(384, dtype=np.float32)

        cleaned = text.strip()
        with self._cache_lock:
            if cleaned in self._embedding_cache:
                return self._embedding_cache[cleaned]

        self._ensure_model_loaded()
        if self._model is None:
            # Fallback zero vector
            return np.zeros(384, dtype=np.float32)

        emb = self._model.encode(
            cleaned,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        emb_arr = np.asarray(emb, dtype=np.float32)

        with self._cache_lock:
            if len(self._embedding_cache) < 8192:
                self._embedding_cache[cleaned] = emb_arr

        return emb_arr

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into a 2D numpy array of normalized vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Check if all in cache
        cached_vectors = []
        uncached_indices = []
        uncached_texts = []

        with self._cache_lock:
            for i, text in enumerate(texts):
                cleaned = text.strip() if text else ""
                if cleaned in self._embedding_cache:
                    cached_vectors.append((i, self._embedding_cache[cleaned]))
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(cleaned)

        if not uncached_texts:
            # All were cached
            res = np.zeros((len(texts), 384), dtype=np.float32)
            for i, vec in cached_vectors:
                res[i] = vec
            return res

        self._ensure_model_loaded()
        if self._model is None:
            return np.zeros((len(texts), 384), dtype=np.float32)

        new_embeddings = self._model.encode(
            uncached_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        new_embeddings = np.asarray(new_embeddings, dtype=np.float32)

        # Assemble full matrix and update cache
        total = len(texts)
        res = np.zeros((total, 384), dtype=np.float32)
        for i, vec in cached_vectors:
            res[i] = vec

        with self._cache_lock:
            for idx_in_uncached, orig_idx in enumerate(uncached_indices):
                vec = new_embeddings[idx_in_uncached]
                res[orig_idx] = vec
                if len(self._embedding_cache) < 8192 and uncached_texts[idx_in_uncached]:
                    self._embedding_cache[uncached_texts[idx_in_uncached]] = vec

        return res

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts (normalized vectors -> dot product)."""
        if not text_a.strip() or not text_b.strip():
            return 0.0

        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)

        # Since embeddings are L2 normalized, cosine similarity is simply the dot product
        sim = float(np.dot(vec_a, vec_b))
        return max(0.0, min(1.0, sim))

    def batch_similarity(
        self,
        source_units: List[str],
        candidate_units: List[str],
    ) -> np.ndarray:
        """Compute all-pairs cosine similarity matrix between source units and candidate units.

        Returns:
            np.ndarray of shape (len(source_units), len(candidate_units)).
        """
        if not source_units or not candidate_units:
            return np.empty((len(source_units), len(candidate_units)), dtype=np.float32)

        source_mat = self.encode_batch(source_units)
        cand_mat = self.encode_batch(candidate_units)

        # Dot product between normalized matrices: (N, 384) @ (384, M) = (N, M)
        sim_matrix = np.matmul(source_mat, cand_mat.T)
        return np.clip(sim_matrix, 0.0, 1.0)

    def max_similarity_per_source(
        self,
        source_units: List[str],
        candidate_units: List[str],
    ) -> List[float]:
        """For each source unit, find maximum similarity against any candidate unit."""
        if not source_units or not candidate_units:
            return [0.0] * len(source_units)

        sim_mat = self.batch_similarity(source_units, candidate_units)
        max_sims = np.max(sim_mat, axis=1)
        return [float(s) for s in max_sims]


# Singleton instance cached in process
semantic_matcher = SemanticMatcher()
