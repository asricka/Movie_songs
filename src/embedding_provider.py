from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for text."""


@dataclass(frozen=True)
class LocalHashEmbeddingProvider:
    """Deterministic local embedding for offline development."""

    dimensions: int = 8

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for i in range(self.dimensions):
            byte = digest[i]
            values.append(byte / 255.0)
        return values


@dataclass(frozen=True)
class OpenAIEmbeddingProvider:
    """Minimal OpenAI embeddings client using HTTPS directly."""

    api_key: str
    model: str = "text-embedding-3-small"

    def embed(self, text: str) -> List[float]:
        body = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        request = Request(
            url="https://api.openai.com/v1/embeddings",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI embedding request failed: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI embedding request failed: {exc}") from exc

        data = payload.get("data", [])
        if not data:
            raise RuntimeError("OpenAI embedding response missing data")
        embedding = data[0].get("embedding")
        if not embedding:
            raise RuntimeError("OpenAI embedding response missing embedding")
        return embedding


def build_embedding_provider() -> EmbeddingProvider:
    provider_name = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        return OpenAIEmbeddingProvider(api_key=api_key, model=model)

    dimensions = int(os.getenv("LOCAL_EMBEDDING_DIMS", "8"))
    return LocalHashEmbeddingProvider(dimensions=dimensions)
