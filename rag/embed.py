"""Sentence embeddings with all-MiniLM-L6-v2 (384 numbers per text).

The model (~90 MB) downloads from Hugging Face on first use and is cached.
It runs locally on CPU; no API key needed.
"""

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbedder:
    dim = 384

    def __init__(self, model=MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device="cpu")

    def encode(self, texts):
        """(len(texts), 384) float32, unit length."""
        return np.asarray(
            self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                              show_progress_bar=False),
            dtype=np.float32,
        )
