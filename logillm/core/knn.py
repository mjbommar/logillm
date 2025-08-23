"""K-nearest neighbors retrieval for demonstration selection."""

from __future__ import annotations

from typing import Any

from .embedders import Embedder, SimpleEmbedder, batch_cosine_similarity


class KNN:
    """K-nearest neighbors retriever for finding similar examples.

    This class provides efficient similarity-based retrieval of examples
    from a training set, supporting both zero-dependency and LLM-based
    embedding approaches.
    """

    def __init__(
        self,
        k: int,
        trainset: list[dict[str, Any]],
        embedder: Embedder | None = None,
        input_keys: list[str] | None = None,
        text_separator: str = " | ",
    ):
        """Initialize KNN retriever.

        Args:
            k: Number of nearest neighbors to retrieve
            trainset: List of training examples to search through
            embedder: Embedder to use (defaults to SimpleEmbedder for zero deps)
            input_keys: Keys to use from inputs for similarity (if None, uses all)
            text_separator: Separator for joining multiple input fields
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not trainset:
            raise ValueError("trainset cannot be empty")

        self.k = k
        self.trainset = trainset
        self.embedder = embedder or SimpleEmbedder()
        self.input_keys = input_keys
        self.text_separator = text_separator

        # Pre-computed embeddings for training set
        self._train_embeddings: list[list[float]] = []
        self._fitted = False

    def _extract_text(self, example: dict[str, Any]) -> str:
        """Extract text from example for embedding.

        Args:
            example: Example dictionary with 'inputs' key

        Returns:
            Text representation of the example's inputs
        """
        inputs = example.get("inputs", {})
        if not inputs:
            return ""

        # Use specified keys or all input keys
        keys_to_use = self.input_keys or list(inputs.keys())

        # Extract text values
        text_parts = []
        for key in keys_to_use:
            if key in inputs:
                value = inputs[key]
                if isinstance(value, (str, int, float)):
                    text_parts.append(f"{key}: {value}")
                else:
                    # Handle complex values by converting to string
                    text_parts.append(f"{key}: {str(value)}")

        return self.text_separator.join(text_parts)

    async def _fit_embeddings(self) -> None:
        """Pre-compute embeddings for training set."""
        if self._fitted:
            return

        # Extract texts from training examples
        train_texts = [self._extract_text(example) for example in self.trainset]

        # Compute embeddings
        self._train_embeddings = await self.embedder.embed(train_texts)
        self._fitted = True

    async def retrieve(self, **query_inputs: Any) -> list[dict[str, Any]]:
        """Retrieve k nearest neighbors for the given query inputs.

        Args:
            **query_inputs: Query inputs to find similar examples for

        Returns:
            List of k most similar examples from trainset
        """
        # Ensure embeddings are fitted
        await self._fit_embeddings()

        # Create query example and extract text
        query_example = {"inputs": query_inputs}
        query_text = self._extract_text(query_example)

        # Embed query
        query_embeddings = await self.embedder.embed([query_text])
        if not query_embeddings:
            return []

        query_vec = query_embeddings[0]

        # Compute similarities with all training examples
        similarities = batch_cosine_similarity(query_vec, self._train_embeddings)

        # Get indices of top k similar examples
        indexed_similarities = [(sim, idx) for idx, sim in enumerate(similarities)]
        indexed_similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top k examples
        top_k = min(self.k, len(indexed_similarities))
        result = []

        for sim, idx in indexed_similarities[:top_k]:
            example = self.trainset[idx].copy()
            # Add similarity score to metadata
            if "metadata" not in example:
                example["metadata"] = {}
            example["metadata"]["similarity"] = sim
            example["metadata"]["retrieval_rank"] = len(result)
            result.append(example)

        return result

    def retrieve_sync(self, **query_inputs: Any) -> list[dict[str, Any]]:
        """Synchronous version of retrieve for convenience."""
        import asyncio

        return asyncio.run(self.retrieve(**query_inputs))

    async def __call__(self, **query_inputs: Any) -> list[dict[str, Any]]:
        """Callable interface for compatibility with DSPy patterns."""
        return await self.retrieve(**query_inputs)

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get statistics about the embeddings."""
        if not self._fitted:
            return {"fitted": False}

        if not self._train_embeddings:
            return {"fitted": True, "num_examples": 0, "embedding_dim": 0}

        embedding_dim = len(self._train_embeddings[0]) if self._train_embeddings else 0

        return {
            "fitted": True,
            "num_examples": len(self._train_embeddings),
            "embedding_dim": embedding_dim,
            "embedder_type": type(self.embedder).__name__,
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_embedding_stats()
        return (
            f"KNN(k={self.k}, trainset_size={len(self.trainset)}, "
            f"fitted={stats['fitted']}, embedder={type(self.embedder).__name__})"
        )


__all__ = ["KNN"]
