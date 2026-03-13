"""
Embedding providers for the CVE search POC.

Two providers:
  - EncoderfileEmbedder: calls a running Encoderfile server (the real integration)
  - FastEmbedEmbedder: uses Qdrant's FastEmbed library (fallback for demo without Encoderfile)

Both implement the same interface so the rest of the app doesn't care which one is active.
"""

import requests
import numpy as np
from typing import Protocol


class Embedder(Protocol):
    """Common interface for embedding providers."""

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...


class EncoderfileEmbedder:
    """
    Generates embeddings via a running Encoderfile server.

    Encoderfile packages transformer encoders into single-binary executables.
    No Python runtime, no dependencies, no network calls — just a fast binary
    serving embeddings over HTTP.

    Start it with:
        ./all-MiniLM-L6-v2.encoderfile serve --port 8080

    This is the integration we're pitching to Qdrant: a zero-dependency
    local embedding provider that pairs with their vector database.
    """

    def __init__(self, url: str = "http://localhost:8080"):
        self.url = url.rstrip("/")
        self._dimension = None

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.url}/predict",
            json={"inputs": texts},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()["results"]
        embeddings = []
        for result in results:
            token_vecs = np.array([e["embedding"] for e in result["embeddings"]])
            embeddings.append(token_vecs.mean(axis=0).tolist())
        return embeddings

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            test = self.embed("test")
            self._dimension = len(test)
        return self._dimension

    def health_check(self) -> bool:
        """Check if the Encoderfile server is reachable."""
        try:
            response = requests.get(f"{self.url}/health", timeout=5)
            return response.ok
        except (requests.ConnectionError, requests.Timeout):
            return False


class FastEmbedEmbedder:
    """
    Fallback embedder using Qdrant's FastEmbed library.

    This lets the POC run end-to-end without Encoderfile for demo purposes.
    In the real integration, this would be replaced by EncoderfileEmbedder.
    Uses the same model (all-MiniLM-L6-v2) so results are comparable.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self._dimension = None

    def embed(self, text: str) -> list[float]:
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            test = self.embed("test")
            self._dimension = len(test)
        return self._dimension


def get_embedder(prefer_encoderfile: bool = True) -> Embedder:
    """
    Returns the best available embedder.
    Tries Encoderfile first if preferred, falls back to FastEmbed.
    """
    if prefer_encoderfile:
        ef = EncoderfileEmbedder()
        if ef.health_check():
            print("✓ Using Encoderfile (localhost:8080)")
            return ef
        else:
            print("⚠ Encoderfile not running, falling back to FastEmbed")

    print("✓ Using FastEmbed (all-MiniLM-L6-v2)")
    return FastEmbedEmbedder()
