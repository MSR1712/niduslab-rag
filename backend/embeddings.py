"""Local, free sentence-embedding model (no API key / rate limits needed).
Kept separate from the LLM so retrieval works even if the LLM quota is used
up, and so latency numbers in the eval harness are meaningful.
"""
from __future__ import annotations

from typing import List
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _get_model():
    # Imported lazily so the rest of the app can be imported/tested without
    # pulling in torch immediately.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
