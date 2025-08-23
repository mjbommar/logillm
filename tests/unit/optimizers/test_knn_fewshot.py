"""Unit tests for KNNFewShot optimizer."""

from unittest.mock import AsyncMock, Mock

import pytest

from logillm.core.embedders import SimpleEmbedder
from logillm.core.knn import KNN
from logillm.core.modules import Module, Parameter
from logillm.core.types import OptimizationResult
from logillm.optimizers.knn_fewshot import KNNFewShot, KNNFewShotConfig


class MockModule(Module):
    """Mock module for testing."""

    def __init__(self, name: str = "mock_module"):
        super().__init__()
        self.name = name
        self.demo_manager = Mock()
        self.demo_manager.clear = Mock()
        self.demo_manager.add = Mock()
        self.parameters = {}
        self._forward_calls = []

    async def forward(self, **inputs):
        self._forward_calls.append(inputs)
        # Mock successful prediction
        from logillm.core.types import Prediction

        return Prediction(
            outputs={"answer": f"mock answer for {inputs}"},
            success=True,
            metadata={"model": self.name},
        )


class MockEmbedder:
    """Mock embedder for testing."""

    def __init__(self, embedding_dim: int = 3):
        self.embedding_dim = embedding_dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Create simple deterministic embeddings
        embeddings = []
        for text in texts:
            h = hash(text) % 100
            embedding = [
                float(h % 10) / 10.0,
                float((h // 10) % 10) / 10.0,
                float(h // 100) / 100.0,
            ]
            embeddings.append(embedding[: self.embedding_dim])
        return embeddings


def dummy_metric(prediction: dict, expected: dict) -> float:
    """Dummy metric for testing."""
    return 0.8  # Return fixed score


class TestKNNFewShotConfig:
    """Test KNNFewShotConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KNNFewShotConfig()

        assert config.k == 3
        assert config.embedder_type == "simple"
        assert config.input_keys is None
        assert config.text_separator == " | "
        assert config.bootstrap_fallback is True
        assert config.fallback_bootstrap_demos == 2
        assert config.min_similarity == 0.0
        assert config.max_similarity == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = KNNFewShotConfig(
            k=5,
            embedder_type="llm",
            input_keys=["question", "context"],
            bootstrap_fallback=False,
            min_similarity=0.3,
        )

        assert config.k == 5
        assert config.embedder_type == "llm"
        assert config.input_keys == ["question", "context"]
        assert config.bootstrap_fallback is False
        assert config.min_similarity == 0.3


class TestKNNFewShot:
    """Test KNNFewShot optimizer."""

    def test_init_basic(self):
        """Test basic initialization."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]

        optimizer = KNNFewShot(k=2, trainset=trainset, metric=dummy_metric)

        assert optimizer.config.k == 2
        assert len(optimizer.trainset) == 2
        assert optimizer.metric == dummy_metric
        assert isinstance(optimizer.embedder, SimpleEmbedder)  # Default embedder
        assert isinstance(optimizer.knn, KNN)

    def test_init_with_custom_embedder(self):
        """Test initialization with custom embedder."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        embedder = MockEmbedder()

        optimizer = KNNFewShot(k=1, trainset=trainset, embedder=embedder, metric=dummy_metric)

        assert optimizer.embedder is embedder
        assert optimizer.knn.embedder is embedder

    def test_init_with_config(self):
        """Test initialization with custom config."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        config = KNNFewShotConfig(k=5, bootstrap_fallback=False, input_keys=["question"])

        optimizer = KNNFewShot(
            k=1,  # Should be overridden by config
            trainset=trainset,
            config=config,
            metric=dummy_metric,
        )

        assert optimizer.config.k == 5  # From config
        assert optimizer.config.bootstrap_fallback is False
        assert optimizer.config.input_keys == ["question"]

    def test_init_validation(self):
        """Test initialization validation."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]

        # Invalid k
        with pytest.raises(ValueError, match="k must be positive"):
            KNNFewShot(k=0, trainset=trainset, metric=dummy_metric)

        # Empty trainset
        with pytest.raises(ValueError, match="trainset cannot be empty"):
            KNNFewShot(k=1, trainset=[], metric=dummy_metric)

        # Missing metric with bootstrap fallback
        with pytest.raises(ValueError, match="metric is required when bootstrap_fallback=True"):
            KNNFewShot(k=1, trainset=trainset, bootstrap_fallback=True)

    def test_init_without_bootstrap_fallback(self):
        """Test initialization without bootstrap fallback."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]

        # Should work without metric when bootstrap_fallback=False
        optimizer = KNNFewShot(k=1, trainset=trainset, bootstrap_fallback=False)

        assert optimizer.metric is None
        assert optimizer.bootstrap_optimizer is None

    @pytest.mark.asyncio
    async def test_create_dynamic_module(self):
        """Test creation of dynamic module."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]
        embedder = MockEmbedder()

        optimizer = KNNFewShot(k=2, trainset=trainset, embedder=embedder, metric=dummy_metric)
        module = MockModule()

        dynamic_module = await optimizer._create_dynamic_module(module, optimizer.knn)

        assert hasattr(dynamic_module, "_knn_retriever")
        assert hasattr(dynamic_module, "_original_forward")
        assert dynamic_module.forward != module.forward  # Should be wrapped

    @pytest.mark.asyncio
    async def test_dynamic_module_forward(self):
        """Test dynamic module forward pass."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]
        embedder = MockEmbedder()

        optimizer = KNNFewShot(k=2, trainset=trainset, embedder=embedder, metric=dummy_metric)
        module = MockModule()

        dynamic_module = await optimizer._create_dynamic_module(module, optimizer.knn)

        # Test forward pass
        result = await dynamic_module.forward(question="What is 1+1?")

        # Should have called demo_manager methods
        dynamic_module.demo_manager.clear.assert_called_once()
        assert dynamic_module.demo_manager.add.call_count > 0

        # Should have added dynamic_demonstrations parameter
        assert "dynamic_demonstrations" in dynamic_module.parameters
        demo_param = dynamic_module.parameters["dynamic_demonstrations"]
        assert isinstance(demo_param, Parameter)
        assert demo_param.learnable is False
        assert demo_param.metadata["type"] == "dynamic_demonstrations"

        # Should have returned a result
        assert result.success

    @pytest.mark.asyncio
    async def test_dynamic_module_forward_error_handling(self):
        """Test dynamic module error handling."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]

        # Create KNN that will fail
        mock_knn = Mock()
        mock_knn.retrieve = AsyncMock(side_effect=Exception("Retrieval failed"))

        optimizer = KNNFewShot(k=1, trainset=trainset, metric=dummy_metric)
        module = MockModule()

        dynamic_module = await optimizer._create_dynamic_module(module, mock_knn)

        # Should still work (fallback to original forward)
        result = await dynamic_module.forward(question="test")
        assert result.success

    @pytest.mark.asyncio
    async def test_optimize_basic(self):
        """Test basic optimization flow."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]
        dataset = [
            {"inputs": {"question": "What is 1+1?"}, "outputs": {"answer": "2"}},
        ]
        embedder = MockEmbedder()

        optimizer = KNNFewShot(k=2, trainset=trainset, embedder=embedder, metric=dummy_metric)
        module = MockModule()

        # Mock evaluate method
        optimizer.evaluate = AsyncMock(return_value=(0.7, None))

        result = await optimizer.optimize(module, dataset)

        assert isinstance(result, OptimizationResult)
        assert result.optimized_module is not None
        assert result.best_score == 0.7
        assert result.improvement == 0.0  # 0.7 - 0.7 (baseline same as final)
        assert result.iterations == 1
        assert "k" in result.metadata
        assert "retrieval_set_size" in result.metadata
        assert result.metadata["similarity_based"] is True
        assert result.metadata["dynamic_demonstrations"] is True

    @pytest.mark.asyncio
    async def test_optimize_with_improvement(self):
        """Test optimization with improvement."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        dataset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]

        optimizer = KNNFewShot(k=1, trainset=trainset, metric=dummy_metric)
        module = MockModule()

        # Mock evaluate to show improvement
        optimizer.evaluate = AsyncMock(side_effect=[(0.6, None), (0.8, None)])  # baseline, final

        result = await optimizer.optimize(module, dataset)

        assert result.improvement == pytest.approx(0.2)  # 0.8 - 0.6
        assert result.best_score == 0.8

    @pytest.mark.asyncio
    async def test_optimize_with_bootstrap_fallback(self):
        """Test optimization with bootstrap fallback."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        dataset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]

        optimizer = KNNFewShot(k=1, trainset=trainset, metric=dummy_metric, bootstrap_fallback=True)
        module = MockModule()

        # Mock evaluate to trigger bootstrap fallback (small improvement)
        optimizer.evaluate = AsyncMock(
            side_effect=[
                (0.7, None),  # baseline
                (0.71, None),  # dynamic (small improvement, triggers fallback)
                (0.85, None),  # hybrid (better)
            ]
        )

        # Mock bootstrap optimizer
        mock_bootstrap_result = OptimizationResult(
            optimized_module=MockModule("bootstrap"),
            improvement=0.1,
            iterations=1,
            best_score=0.8,
            optimization_time=1.0,
            metadata={},
        )
        optimizer.bootstrap_optimizer = Mock()
        optimizer.bootstrap_optimizer.optimize = AsyncMock(return_value=mock_bootstrap_result)

        result = await optimizer.optimize(module, dataset)

        # Should have used hybrid approach
        assert result.improvement > 0.1  # Better than just KNN
        optimizer.bootstrap_optimizer.optimize.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_bootstrap_fallback_failure(self):
        """Test optimization when bootstrap fallback fails."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        dataset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]

        optimizer = KNNFewShot(k=1, trainset=trainset, metric=dummy_metric, bootstrap_fallback=True)
        module = MockModule()

        # Mock evaluate to trigger bootstrap fallback
        optimizer.evaluate = AsyncMock(side_effect=[(0.7, None), (0.71, None)])

        # Mock bootstrap optimizer to fail
        optimizer.bootstrap_optimizer = Mock()
        optimizer.bootstrap_optimizer.optimize = AsyncMock(
            side_effect=Exception("Bootstrap failed")
        )

        result = await optimizer.optimize(module, dataset)

        # Should still return KNN result
        assert result.improvement == pytest.approx(0.01)  # 0.71 - 0.7
        assert result.best_score == 0.71

    @pytest.mark.asyncio
    async def test_optimize_with_validation_set(self):
        """Test optimization with separate validation set."""
        trainset = [{"inputs": {"question": "train"}, "outputs": {"answer": "train"}}]
        dataset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        validation_set = [{"inputs": {"question": "valid"}, "outputs": {"answer": "valid"}}]

        optimizer = KNNFewShot(
            k=1, trainset=trainset, metric=dummy_metric, bootstrap_fallback=False
        )  # Disable fallback
        module = MockModule()

        optimizer.evaluate = AsyncMock(return_value=(0.7, None))

        await optimizer.optimize(module, dataset, validation_set)

        # Should evaluate on validation set (baseline + final)
        assert optimizer.evaluate.call_count == 2  # baseline and final
        call_args = optimizer.evaluate.call_args_list
        assert call_args[0][0][1] == validation_set  # baseline eval
        assert call_args[1][0][1] == validation_set  # final eval

    @pytest.mark.asyncio
    async def test_optimize_empty_trainset_uses_dataset(self):
        """Test optimization when trainset is empty uses dataset."""
        # Initialize with empty trainset then set it to dataset during optimize
        trainset = [{"inputs": {"question": "placeholder"}, "outputs": {"answer": "placeholder"}}]
        dataset = [
            {"inputs": {"question": "What is 1+1?"}, "outputs": {"answer": "2"}},
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
        ]

        optimizer = KNNFewShot(k=1, trainset=trainset, metric=dummy_metric)
        # Override trainset to be empty to test fallback
        optimizer.trainset = []

        module = MockModule()
        optimizer.evaluate = AsyncMock(return_value=(0.7, None))

        result = await optimizer.optimize(module, dataset)

        # Should use dataset as retrieval set
        assert result.metadata["retrieval_set_size"] == len(dataset)

    def test_str_representation(self):
        """Test string representation."""
        trainset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]
        optimizer = KNNFewShot(k=2, trainset=trainset, metric=dummy_metric)

        # Should not raise an exception
        str_repr = str(optimizer)
        assert isinstance(str_repr, str)

    @pytest.mark.asyncio
    async def test_knn_embedding_fitting(self):
        """Test that KNN embeddings are fitted during optimization."""
        trainset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 3+3?"}, "outputs": {"answer": "6"}},
        ]
        dataset = [{"inputs": {"question": "What is 1+1?"}, "outputs": {"answer": "2"}}]
        embedder = MockEmbedder()

        optimizer = KNNFewShot(k=2, trainset=trainset, embedder=embedder, metric=dummy_metric)
        module = MockModule()
        optimizer.evaluate = AsyncMock(return_value=(0.7, None))

        # KNN should not be fitted initially
        assert not optimizer.knn._fitted

        await optimizer.optimize(module, dataset)

        # KNN should be fitted after optimization
        assert optimizer.knn._fitted

    @pytest.mark.asyncio
    async def test_custom_input_keys(self):
        """Test optimization with custom input keys."""
        trainset = [
            {
                "inputs": {"question": "math", "context": "school", "noise": "ignore"},
                "outputs": {"answer": "test"},
            },
        ]
        dataset = [{"inputs": {"question": "test"}, "outputs": {"answer": "test"}}]

        optimizer = KNNFewShot(
            k=1, trainset=trainset, input_keys=["question", "context"], metric=dummy_metric
        )
        module = MockModule()
        optimizer.evaluate = AsyncMock(return_value=(0.7, None))

        result = await optimizer.optimize(module, dataset)

        # Should use custom input keys
        assert optimizer.knn.input_keys == ["question", "context"]
        assert result.metadata["config"]["input_keys"] == ["question", "context"]
