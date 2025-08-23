"""Text embedding abstraction with zero-dependency implementations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any


class Embedder(ABC):
    """Abstract base class for text embedding."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vectors.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per text)
        """
        ...

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous version of embed for convenience."""
        import asyncio

        return asyncio.run(self.embed(texts))


class SimpleEmbedder(Embedder):
    """Simple TF-IDF based embedder using only standard library.

    This provides a zero-dependency embedding solution for basic similarity
    matching. While not as sophisticated as neural embeddings, it works
    reasonably well for many applications and maintains LogiLLM's principle
    of minimal dependencies.
    """

    def __init__(
        self,
        max_features: int = 1000,
        min_df: int = 1,
        max_df: float = 0.95,
        normalize: bool = True,
        lowercase: bool = True,
        stop_words: set[str] | None = None,
    ):
        """Initialize SimpleEmbedder.

        Args:
            max_features: Maximum number of features (vocabulary size)
            min_df: Minimum document frequency for a term to be included
            max_df: Maximum document frequency (as fraction) for a term to be included
            normalize: Whether to normalize vectors to unit length
            lowercase: Whether to convert text to lowercase
            stop_words: Set of stop words to ignore
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.normalize = normalize
        self.lowercase = lowercase
        self.stop_words = stop_words or self._default_stop_words()

        # Vocabulary and IDF scores (fitted on training data)
        self.vocabulary_: dict[str, int] = {}
        self.idf_: list[float] = []
        self._fitted = False

    def _default_stop_words(self) -> set[str]:
        """Default English stop words."""
        return {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "this",
            "but",
            "they",
            "have",
            "had",
            "what",
            "said",
            "each",
            "which",
            "do",
            "how",
            "their",
            "if",
        }

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        if self.lowercase:
            text = text.lower()

        # Simple word splitting (alphanumeric + basic punctuation handling)
        import re

        tokens = re.findall(r"\w+", text)

        # Remove stop words
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]

        return tokens

    def _fit_vocabulary(self, texts: list[str]) -> None:
        """Fit vocabulary and compute IDF scores."""
        if self._fitted:
            return

        # Tokenize all texts
        tokenized_docs = [self._tokenize(text) for text in texts]

        # Count document frequencies
        doc_frequencies: dict[str, int] = {}
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_frequencies[token] = doc_frequencies.get(token, 0) + 1

        # Filter by min/max document frequency
        n_docs = len(texts)
        min_count = max(1, self.min_df)
        max_count = int(self.max_df * n_docs) if self.max_df < 1.0 else n_docs

        # Build vocabulary
        vocab_items = [
            (token, count)
            for token, count in doc_frequencies.items()
            if min_count <= count <= max_count
        ]

        # Sort by frequency and take top features
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        vocab_items = vocab_items[: self.max_features]

        # Create vocabulary mapping
        self.vocabulary_ = {token: idx for idx, (token, _) in enumerate(vocab_items)}

        # Compute IDF scores: idf = log(n_docs / df)
        self.idf_ = []
        for token, _ in vocab_items:
            df = doc_frequencies[token]
            idf = math.log(n_docs / df)
            self.idf_.append(idf)

        self._fitted = True

    def _vectorize(self, text: str) -> list[float]:
        """Convert text to TF-IDF vector."""
        if not self._fitted:
            raise ValueError("Embedder not fitted. Call embed() with training data first.")

        tokens = self._tokenize(text)
        tf_counts = Counter(tokens)

        # Create TF-IDF vector
        vector = [0.0] * len(self.vocabulary_)
        total_tokens = len(tokens)

        for token, count in tf_counts.items():
            if token in self.vocabulary_:
                idx = self.vocabulary_[token]
                tf = count / total_tokens if total_tokens > 0 else 0.0
                tfidf = tf * self.idf_[idx]
                vector[idx] = tfidf

        # Normalize to unit length
        if self.normalize:
            norm = math.sqrt(sum(x * x for x in vector))
            if norm > 0:
                vector = [x / norm for x in vector]

        return vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into TF-IDF vectors.

        Args:
            texts: List of texts to embed

        Returns:
            List of TF-IDF vectors
        """
        if not texts:
            return []

        # Fit vocabulary on the first call
        if not self._fitted:
            self._fit_vocabulary(texts)

        # Vectorize all texts
        vectors = [self._vectorize(text) for text in texts]
        return vectors


class LLMEmbedder(Embedder):
    """Use LLM provider to generate embeddings (if supported)."""

    def __init__(self, provider: Any, model: str | None = None, **kwargs: Any):
        """Initialize LLMEmbedder.

        Args:
            provider: LLM provider with embed() method
            model: Optional model override for embeddings
            **kwargs: Additional embedding parameters
        """
        self.provider = provider
        self.model = model
        self.kwargs = kwargs

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using LLM provider.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors from the provider

        Raises:
            ValueError: If provider doesn't support embedding
        """
        if not hasattr(self.provider, "embed"):
            raise ValueError(f"Provider {self.provider.name} does not support embedding")

        # Use provider's embed method
        embeddings = await self.provider.embed(texts, model=self.model, **self.kwargs)
        return embeddings


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score [-1, 1]
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")

    if not vec1 or not vec2:
        return 0.0

    # Compute dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Compute magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    # Avoid division by zero
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def batch_cosine_similarity(query_vec: list[float], vectors: list[list[float]]) -> list[float]:
    """Compute cosine similarity between query vector and batch of vectors.

    Args:
        query_vec: Query vector to compare against
        vectors: List of vectors to compare with query

    Returns:
        List of cosine similarity scores
    """
    return [cosine_similarity(query_vec, vec) for vec in vectors]


__all__ = [
    "Embedder",
    "SimpleEmbedder",
    "LLMEmbedder",
    "cosine_similarity",
    "batch_cosine_similarity",
]
