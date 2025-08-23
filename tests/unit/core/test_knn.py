"""Unit tests for KNN module."""

import pytest

from logillm.core.embedders import SimpleEmbedder
from logillm.core.knn import KNN


class MockEmbedder:
    """Mock embedder for testing KNN."""

    def __init__(self, embedding_dim: int = 3):
        self.embedding_dim = embedding_dim
        self.embeddings = {}  # Cache for deterministic results

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            if text not in self.embeddings:
                # Create deterministic embedding based on text
                h = hash(text) % 1000
                embedding = []
                for i in range(self.embedding_dim):
                    embedding.append(float(((h // (10**i)) % 10) / 10.0))
                self.embeddings[text] = embedding
            embeddings.append(self.embeddings[text])
        return embeddings


class TestKNN:
    """Test KNN retrieval functionality."""

    def test_init(self):
        """Test KNN initialization."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]
        embedder = MockEmbedder()

        knn = KNN(k=2, trainset=trainset, embedder=embedder)

        assert knn.k == 2
        assert len(knn.trainset) == 2
        assert knn.embedder is embedder
        assert not knn._fitted

    def test_init_validation(self):
        """Test KNN initialization validation."""
        trainset = [{"inputs": {"q": "test"}, "outputs": {"a": "test"}}]

        # Test invalid k
        with pytest.raises(ValueError, match="k must be positive"):
            KNN(k=0, trainset=trainset)

        with pytest.raises(ValueError, match="k must be positive"):
            KNN(k=-1, trainset=trainset)

        # Test empty trainset
        with pytest.raises(ValueError, match="trainset cannot be empty"):
            KNN(k=1, trainset=[])

    def test_extract_text_simple(self):
        """Test text extraction from examples."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        knn = KNN(k=1, trainset=trainset)

        example = {"inputs": {"question": "What is 2+2?"}}
        text = knn._extract_text(example)

        assert text == "question: What is 2+2?"

    def test_extract_text_multiple_fields(self):
        """Test text extraction with multiple input fields."""
        trainset = [{"inputs": {"q": "test"}, "outputs": {"a": "test"}}]
        knn = KNN(k=1, trainset=trainset, text_separator=" || ")

        example = {"inputs": {"question": "What is 2+2?", "context": "Math problem"}}
        text = knn._extract_text(example)

        assert "question: What is 2+2?" in text
        assert "context: Math problem" in text
        assert "||" in text

    def test_extract_text_with_input_keys(self):
        """Test text extraction with specific input keys."""
        trainset = [{"inputs": {"q": "test"}, "outputs": {"a": "test"}}]
        knn = KNN(k=1, trainset=trainset, input_keys=["question"])

        example = {"inputs": {"question": "What is 2+2?", "context": "ignore this"}}
        text = knn._extract_text(example)

        assert "question: What is 2+2?" in text
        assert "context" not in text

    def test_extract_text_empty_inputs(self):
        """Test text extraction with empty inputs."""
        trainset = [{"inputs": {"q": "test"}, "outputs": {"a": "test"}}]
        knn = KNN(k=1, trainset=trainset)

        example = {"inputs": {}}
        text = knn._extract_text(example)

        assert text == ""

    def test_extract_text_missing_inputs(self):
        """Test text extraction with missing inputs key."""
        trainset = [{"inputs": {"q": "test"}, "outputs": {"a": "test"}}]
        knn = KNN(k=1, trainset=trainset)

        example = {}
        text = knn._extract_text(example)

        assert text == ""

    def test_extract_text_complex_values(self):
        """Test text extraction with complex values."""
        trainset = [{"inputs": {"q": "test"}, "outputs": {"a": "test"}}]
        knn = KNN(k=1, trainset=trainset)

        example = {"inputs": {"number": 42, "data": {"nested": "value"}}}
        text = knn._extract_text(example)

        assert "number: 42" in text
        assert "data:" in text

    @pytest.mark.asyncio
    async def test_fit_embeddings(self):
        """Test embedding fitting."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]
        embedder = MockEmbedder()
        knn = KNN(k=2, trainset=trainset, embedder=embedder)

        await knn._fit_embeddings()

        assert knn._fitted
        assert len(knn._train_embeddings) == 2
        assert all(len(emb) == 3 for emb in knn._train_embeddings)

    @pytest.mark.asyncio
    async def test_fit_embeddings_idempotent(self):
        """Test that fitting embeddings is idempotent."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        embedder = MockEmbedder()
        knn = KNN(k=1, trainset=trainset, embedder=embedder)

        await knn._fit_embeddings()
        first_embeddings = knn._train_embeddings.copy()

        await knn._fit_embeddings()  # Should not recompute

        assert knn._train_embeddings == first_embeddings

    @pytest.mark.asyncio
    async def test_retrieve_basic(self):
        """Test basic retrieval functionality."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
            {"inputs": {"question": "Hello world"}, "outputs": {"answer": "Hi"}},
        ]
        embedder = MockEmbedder()
        knn = KNN(k=2, trainset=trainset, embedder=embedder)

        results = await knn.retrieve(question="What is 1+1?")

        assert len(results) == 2
        for result in results:
            assert "inputs" in result
            assert "outputs" in result
            assert "metadata" in result
            assert "similarity" in result["metadata"]
            assert "retrieval_rank" in result["metadata"]

    @pytest.mark.asyncio
    async def test_retrieve_k_larger_than_trainset(self):
        """Test retrieval when k > trainset size."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
        ]
        embedder = MockEmbedder()
        knn = KNN(k=5, trainset=trainset, embedder=embedder)

        results = await knn.retrieve(question="What is 1+1?")

        assert len(results) == 1  # Should return all available

    @pytest.mark.asyncio
    async def test_retrieve_similarity_ranking(self):
        """Test that results are ranked by similarity."""
        trainset = [
            {"inputs": {"question": "math problem 2+2"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "totally different topic"}, "outputs": {"answer": "other"}},
            {"inputs": {"question": "another math problem 3+3"}, "outputs": {"answer": "6"}},
        ]
        embedder = MockEmbedder()
        knn = KNN(k=3, trainset=trainset, embedder=embedder)

        results = await knn.retrieve(question="math problem 1+1")

        assert len(results) == 3
        # Results should be ordered by similarity (decreasing)
        similarities = [r["metadata"]["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

        # Retrieval rank should be in order
        ranks = [r["metadata"]["retrieval_rank"] for r in results]
        assert ranks == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_retrieve_with_metadata_preserved(self):
        """Test that original metadata is preserved in results."""
        trainset = [
            {
                "inputs": {"question": "test"},
                "outputs": {"answer": "test"},
                "metadata": {"original": "data", "score": 0.9},
            },
        ]
        embedder = MockEmbedder()
        knn = KNN(k=1, trainset=trainset, embedder=embedder)

        results = await knn.retrieve(question="test query")

        assert len(results) == 1
        result = results[0]
        assert result["metadata"]["original"] == "data"
        assert result["metadata"]["score"] == 0.9
        assert "similarity" in result["metadata"]  # Added by KNN
        assert "retrieval_rank" in result["metadata"]  # Added by KNN

    def test_retrieve_sync(self):
        """Test synchronous retrieve wrapper."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        embedder = MockEmbedder()
        knn = KNN(k=1, trainset=trainset, embedder=embedder)

        results = knn.retrieve_sync(question="test query")

        assert len(results) == 1
        assert "inputs" in results[0]
        assert "outputs" in results[0]

    @pytest.mark.asyncio
    async def test_callable_interface(self):
        """Test that KNN can be called directly."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        embedder = MockEmbedder()
        knn = KNN(k=1, trainset=trainset, embedder=embedder)

        results = await knn(question="test query")

        assert len(results) == 1
        assert "inputs" in results[0]

    def test_get_embedding_stats_unfitted(self):
        """Test embedding stats before fitting."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        knn = KNN(k=1, trainset=trainset)

        stats = knn.get_embedding_stats()

        assert stats["fitted"] is False

    @pytest.mark.asyncio
    async def test_get_embedding_stats_fitted(self):
        """Test embedding stats after fitting."""
        trainset = [
            {"inputs": {"question": "test1"}, "outputs": {"answer": "test1"}},
            {"inputs": {"question": "test2"}, "outputs": {"answer": "test2"}},
        ]
        embedder = MockEmbedder(embedding_dim=5)
        knn = KNN(k=1, trainset=trainset, embedder=embedder)

        await knn._fit_embeddings()
        stats = knn.get_embedding_stats()

        assert stats["fitted"] is True
        assert stats["num_examples"] == 2
        assert stats["embedding_dim"] == 5
        assert stats["embedder_type"] == "MockEmbedder"

    def test_get_embedding_stats_empty_embeddings(self):
        """Test embedding stats with empty embeddings."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        embedder = MockEmbedder()
        knn = KNN(k=1, trainset=trainset, embedder=embedder)

        knn._fitted = True
        knn._train_embeddings = []

        stats = knn.get_embedding_stats()

        assert stats["fitted"] is True
        assert stats["num_examples"] == 0
        assert stats["embedding_dim"] == 0

    def test_repr(self):
        """Test string representation."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        embedder = SimpleEmbedder()
        knn = KNN(k=3, trainset=trainset, embedder=embedder)

        repr_str = repr(knn)

        assert "KNN(" in repr_str
        assert "k=3" in repr_str
        assert "trainset_size=1" in repr_str
        assert "fitted=False" in repr_str
        assert "SimpleEmbedder" in repr_str

    @pytest.mark.asyncio
    async def test_default_embedder(self):
        """Test that default embedder is used when none provided."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        knn = KNN(k=1, trainset=trainset)  # No embedder provided

        assert isinstance(knn.embedder, SimpleEmbedder)

        # Should work with default embedder
        results = await knn.retrieve(question="test query")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_custom_input_keys(self):
        """Test retrieval with custom input keys."""
        trainset = [
            {
                "inputs": {"question": "math", "context": "education", "irrelevant": "noise"},
                "outputs": {"answer": "test"},
            },
        ]
        knn = KNN(k=1, trainset=trainset, input_keys=["question", "context"])

        # Fit embeddings and check what text was used
        await knn._fit_embeddings()

        # Should have used only question and context for similarity
        results = await knn.retrieve(question="math", context="education")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_text_separator(self):
        """Test custom text separator."""
        trainset = [
            {"inputs": {"field1": "value1", "field2": "value2"}, "outputs": {"answer": "test"}},
        ]
        knn = KNN(k=1, trainset=trainset, text_separator=" ### ")

        example = {"inputs": {"field1": "value1", "field2": "value2"}}
        text = knn._extract_text(example)

        assert "###" in text
        assert "field1: value1" in text
        assert "field2: value2" in text
