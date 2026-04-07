"""Embedding utilities using sentence-transformers with CUDA auto-selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import SETTINGS


class EmbeddingEncoder:
    """Wrap sentence-transformers model for consistent encoding."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or SETTINGS.model.embedding_model
        self.device = SETTINGS.device
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Encode a list of strings into dense embeddings."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return vectors.astype(np.float32)
