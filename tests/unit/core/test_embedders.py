"""Unit tests for embedders module."""

from unittest.mock import AsyncMock, Mock

import pytest

from logillm.core.embedders import (
    Embedder,
    LLMEmbedder,
    SimpleEmbedder,
    batch_cosine_similarity,
    cosine_similarity,
)


class TestCosinesimilarity:
    """Test cosine similarity functions."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity with opposite vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity with different length vectors."""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="Vectors must have same length"):
            cosine_similarity(vec1, vec2)

    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity computation."""
        query = [1.0, 0.0, 0.0]
        vectors = [
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [-1.0, 0.0, 0.0],  # opposite
            [0.5, 0.0, 0.0],  # same direction, different magnitude
        ]

        similarities = batch_cosine_similarity(query, vectors)

        assert len(similarities) == 4
        assert similarities[0] == pytest.approx(1.0)
        assert similarities[1] == pytest.approx(0.0)
        assert similarities[2] == pytest.approx(-1.0)
        assert similarities[3] == pytest.approx(1.0)  # Same direction


class TestSimpleEmbedder:
    """Test SimpleEmbedder TF-IDF implementation."""

    def test_init(self):
        """Test SimpleEmbedder initialization."""
        embedder = SimpleEmbedder()
        assert embedder.max_features == 1000
        assert embedder.normalize is True
        assert embedder.lowercase is True
        assert not embedder._fitted

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        embedder = SimpleEmbedder()
        tokens = embedder._tokenize("Hello world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" not in tokens  # stop word
        assert "are" not in tokens  # stop word
        # Note: 'you' is not in our default stop words, so it should be present
        assert "you" in tokens

    def test_tokenize_with_stop_words(self):
        """Test tokenization with custom stop words."""
        embedder = SimpleEmbedder(stop_words={"hello"})
        tokens = embedder._tokenize("Hello world")
        assert "hello" not in tokens
        assert "world" in tokens

    def test_tokenize_case_sensitive(self):
        """Test tokenization with case sensitivity."""
        embedder = SimpleEmbedder(lowercase=False, stop_words=set())
        tokens = embedder._tokenize("Hello World")
        assert "Hello" in tokens
        assert "World" in tokens
        assert "hello" not in tokens

    @pytest.mark.asyncio
    async def test_embed_simple_texts(self):
        """Test embedding simple texts."""
        embedder = SimpleEmbedder(max_features=10)
        texts = ["hello world", "world peace", "hello peace"]

        embeddings = await embedder.embed(texts)

        assert len(embeddings) == 3
        assert all(len(emb) > 0 for emb in embeddings)

        # All embeddings should have same dimensionality
        assert len({len(emb) for emb in embeddings}) == 1

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embedding empty text list."""
        embedder = SimpleEmbedder()
        embeddings = await embedder.embed([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_embed_similarity(self):
        """Test that similar texts have higher similarity."""
        embedder = SimpleEmbedder()
        texts = ["the cat sat on the mat", "a cat sat on a mat", "dogs run in the park"]

        embeddings = await embedder.embed(texts)

        # Similar texts should have higher similarity
        sim_cat1_cat2 = cosine_similarity(embeddings[0], embeddings[1])
        sim_cat1_dog = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_cat1_cat2 > sim_cat1_dog

    def test_embed_sync(self):
        """Test synchronous embedding wrapper."""
        embedder = SimpleEmbedder()
        texts = ["hello world", "goodbye world"]

        embeddings = embedder.embed_sync(texts)

        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_vocab_fitting(self):
        """Test vocabulary fitting."""
        embedder = SimpleEmbedder(max_features=3, min_df=1)
        texts = ["hello world", "hello python", "world python"]

        await embedder.embed(texts)

        assert embedder._fitted
        # Should fit vocabulary from the texts
        assert len(embedder.vocabulary_) <= 3
        assert len(embedder.idf_) == len(embedder.vocabulary_)

    @pytest.mark.asyncio
    async def test_document_frequency_filtering(self):
        """Test document frequency filtering."""
        embedder = SimpleEmbedder(max_features=10, min_df=2)  # Require term in at least 2 docs
        texts = ["unique term", "common word", "common word", "other common word"]

        await embedder.embed(texts)

        # 'unique' should be filtered out (appears in only 1 doc)
        assert "unique" not in embedder.vocabulary_
        # 'common' should be included (appears in 3 docs)
        assert "common" in embedder.vocabulary_

    @pytest.mark.asyncio
    async def test_normalization(self):
        """Test vector normalization."""
        embedder = SimpleEmbedder(
            normalize=True, max_features=10, stop_words=set(), min_df=1
        )  # No stop words
        texts = [
            "unique special words for testing",
            "more unique words here",
        ]  # Multiple texts for IDF

        embeddings = await embedder.embed(texts)

        # Check that we actually have embeddings
        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings), "Embeddings should not be empty"

        # Test first embedding
        if len(embeddings[0]) > 0:
            # Calculate magnitude
            magnitude = sum(x * x for x in embeddings[0]) ** 0.5

            # If we have any non-zero values, should be normalized to unit length
            total_magnitude = sum(abs(x) for x in embeddings[0])
            if total_magnitude > 0:
                assert magnitude == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_no_normalization(self):
        """Test without vector normalization."""
        embedder = SimpleEmbedder(normalize=False)
        texts = ["hello world test"]

        embeddings = await embedder.embed(texts)

        # Should not be normalized
        magnitude = sum(x * x for x in embeddings[0]) ** 0.5
        assert magnitude != pytest.approx(1.0)


class TestLLMEmbedder:
    """Test LLMEmbedder implementation."""

    @pytest.mark.asyncio
    async def test_embed_with_provider(self):
        """Test embedding with a mock provider."""
        mock_provider = Mock()
        mock_provider.name = "mock_provider"
        mock_provider.embed = AsyncMock(return_value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        embedder = LLMEmbedder(mock_provider)
        texts = ["text1", "text2"]

        embeddings = await embedder.embed(texts)

        assert embeddings == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mock_provider.embed.assert_called_once_with(texts, model=None)

    @pytest.mark.asyncio
    async def test_embed_with_model_override(self):
        """Test embedding with model override."""
        mock_provider = Mock()
        mock_provider.name = "mock_provider"
        mock_provider.embed = AsyncMock(return_value=[[1.0, 2.0]])

        embedder = LLMEmbedder(mock_provider, model="custom-model")
        texts = ["text1"]

        await embedder.embed(texts)

        mock_provider.embed.assert_called_once_with(texts, model="custom-model")

    @pytest.mark.asyncio
    async def test_embed_with_kwargs(self):
        """Test embedding with additional kwargs."""
        mock_provider = Mock()
        mock_provider.name = "mock_provider"
        mock_provider.embed = AsyncMock(return_value=[[1.0, 2.0]])

        embedder = LLMEmbedder(mock_provider, batch_size=10, timeout=30)
        texts = ["text1"]

        await embedder.embed(texts)

        mock_provider.embed.assert_called_once_with(texts, model=None, batch_size=10, timeout=30)

    @pytest.mark.asyncio
    async def test_embed_provider_without_embed(self):
        """Test embedding with provider that doesn't support embedding."""
        mock_provider = Mock(spec=[])  # Empty spec means no methods/attributes
        mock_provider.name = "unsupported_provider"

        embedder = LLMEmbedder(mock_provider)

        # Should raise ValueError before trying to await
        with pytest.raises(
            ValueError, match="Provider unsupported_provider does not support embedding"
        ):
            await embedder.embed(["text1"])

    def test_embed_sync(self):
        """Test synchronous embedding wrapper."""
        mock_provider = Mock()
        mock_provider.name = "mock_provider"
        mock_provider.embed = AsyncMock(return_value=[[1.0, 2.0, 3.0]])

        embedder = LLMEmbedder(mock_provider)
        texts = ["text1"]

        embeddings = embedder.embed_sync(texts)

        assert embeddings == [[1.0, 2.0, 3.0]]


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    def __init__(self, embedding_dim: int = 3):
        self.embedding_dim = embedding_dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        # Return simple hash-based embeddings for testing
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            h = hash(text) % 1000
            embedding = [float(h % 10), float((h // 10) % 10), float((h // 100) % 10)]
            embeddings.append(embedding[: self.embedding_dim])
        return embeddings


class TestEmbedderInterface:
    """Test abstract Embedder interface."""

    @pytest.mark.asyncio
    async def test_mock_embedder(self):
        """Test mock embedder implementation."""
        embedder = MockEmbedder(embedding_dim=2)
        texts = ["hello", "world"]

        embeddings = await embedder.embed(texts)

        assert len(embeddings) == 2
        assert all(len(emb) == 2 for emb in embeddings)
        assert embedder.call_count == 1

    def test_embed_sync_wrapper(self):
        """Test sync wrapper of abstract embedder."""
        embedder = MockEmbedder()
        texts = ["test"]

        embeddings = embedder.embed_sync(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3
