"""Unit tests for evaluation framework."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from logillm.core.types import Prediction
from logillm.evaluate.evaluator import Evaluate, EvaluationResult


@pytest.fixture
def mock_module():
    """Create a mock module for testing."""
    module = MagicMock()

    # Create async forward method
    async def mock_forward(**kwargs):
        # Return different answers based on input
        question = kwargs.get("question", "")
        if "2+2" in question:
            answer = "4"
        elif "capital" in question:
            answer = "Paris"
        elif "Hamlet" in question:
            answer = "William Shakespeare"  # Different from "Shakespeare"
        else:
            answer = "test answer"

        return Prediction(success=True, outputs={"answer": answer}, metadata={})

    module.forward = AsyncMock(side_effect=mock_forward)
    return module


@pytest.fixture
def sample_dataset():
    """Create sample evaluation dataset."""
    return [
        {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"answer": "Paris"}},
        {"inputs": {"question": "Who wrote Hamlet?"}, "outputs": {"answer": "Shakespeare"}},
    ]


@pytest.fixture
def mock_metric():
    """Create a mock metric."""

    # Return different scores for different predictions
    def score_func(pred, ref):
        if pred.get("answer") == ref.get("answer"):
            return 1.0
        return 0.0

    # Create the mock with the callable behavior
    metric = MagicMock(side_effect=score_func)
    metric.__name__ = "MockMetric"
    return metric


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_initialization(self):
        """Test result initialization."""
        result = EvaluationResult(
            score=0.85,
            scores=[0.8, 0.9, 0.85],
            predictions=[{"answer": "test1"}, {"answer": "test2"}, {"answer": "test3"}],
            successes=2,
            failures=1,
            metadata={"test": "value"},
        )

        assert result.score == 0.85
        assert len(result.scores) == 3
        assert len(result.predictions) == 3
        assert result.successes == 2
        assert result.failures == 1
        assert result.metadata["test"] == "value"

    def test_timing_info(self):
        """Test timing information in result."""
        result = EvaluationResult(
            score=0.9,
            scores=[],
            predictions=[],
            successes=1,
            failures=0,
            total_time=2.5,
            avg_time_per_example=2.5,
        )

        assert result.total_time == 2.5
        assert result.avg_time_per_example == 2.5

    def test_string_representation(self):
        """Test string representation."""
        result = EvaluationResult(score=0.75, scores=[], predictions=[], successes=3, failures=1)

        str_repr = str(result)
        assert "0.750" in str_repr
        assert "3/4" in str_repr

    def test_summary(self):
        """Test summary generation."""
        result = EvaluationResult(
            score=0.667,
            scores=[],
            predictions=[],
            successes=2,
            failures=1,
            total_time=1.5,
            avg_time_per_example=0.5,
        )

        summary = result.summary()
        assert "Score: 0.667" in summary
        assert "Successes: 2/3" in summary
        assert "66.7%" in summary


class TestEvaluate:
    """Test Evaluate class."""

    def test_initialization(self, sample_dataset, mock_metric):
        """Test evaluator initialization."""
        evaluator = Evaluate(dataset=sample_dataset, metric=mock_metric)
        assert evaluator.dataset == sample_dataset
        assert evaluator.metric == mock_metric
        assert evaluator.num_threads == 1
        assert evaluator.display_progress is True

        evaluator2 = Evaluate(
            dataset=sample_dataset, metric=mock_metric, num_threads=5, display_progress=False
        )
        assert evaluator2.num_threads == 5
        assert evaluator2.display_progress is False

    @pytest.mark.asyncio
    async def test_evaluate_example(self, mock_module, sample_dataset, mock_metric):
        """Test evaluating a single example."""
        evaluator = Evaluate(dataset=sample_dataset, metric=mock_metric)

        example = sample_dataset[0]

        result = await evaluator._evaluate_example(module=mock_module, example=example)

        assert result["success"] is True
        assert result["score"] == 1.0  # Exact match
        assert result["prediction"]["answer"] == "4"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_evaluate_example_with_error(self, sample_dataset, mock_metric):
        """Test handling errors in single evaluation."""
        evaluator = Evaluate(dataset=sample_dataset, metric=mock_metric)

        # Create module that raises error
        error_module = MagicMock()
        error_module.forward = AsyncMock(side_effect=Exception("Test error"))

        example = sample_dataset[0]

        result = await evaluator._evaluate_example(module=error_module, example=example)

        assert result["success"] is False
        assert result["score"] == 0.0  # Error results in 0 score
        assert result["prediction"] is None
        assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_call_evaluation(self, mock_module, sample_dataset, mock_metric):
        """Test full evaluation run via __call__."""
        evaluator = Evaluate(dataset=sample_dataset, metric=mock_metric, display_progress=False)

        result = await evaluator(mock_module)

        assert isinstance(result, EvaluationResult)
        # First two should match (2+2=4, capital=Paris)
        # Third won't match (Hamlet vs Shakespeare)
        assert result.score == pytest.approx(2 / 3, 0.01)
        # All three predictions succeed (no errors), but only 2 get perfect scores
        assert result.successes == 3  # No errors occurred
        assert result.failures == 0  # No prediction failures
        assert len(result.predictions) == 3

    @pytest.mark.asyncio
    async def test_sequential_evaluate(self, mock_module, sample_dataset, mock_metric):
        """Test sequential evaluation."""
        evaluator = Evaluate(
            dataset=sample_dataset, metric=mock_metric, num_threads=1, display_progress=False
        )

        results = await evaluator._sequential_evaluate(mock_module, sample_dataset)

        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_parallel_evaluate(self, mock_module, sample_dataset, mock_metric):
        """Test parallel evaluation."""
        evaluator = Evaluate(
            dataset=sample_dataset, metric=mock_metric, num_threads=2, display_progress=False
        )

        results = await evaluator._parallel_evaluate(mock_module, sample_dataset)

        assert len(results) == 3
        # Check that results are properly collected
        success_count = sum(1 for r in results if r["success"])
        assert success_count > 0

    @pytest.mark.asyncio
    async def test_empty_dataset(self, mock_module, mock_metric):
        """Test evaluation with empty dataset."""
        evaluator = Evaluate(dataset=[], metric=mock_metric)

        with pytest.raises(ValueError, match="No dataset"):
            await evaluator(mock_module)

    @pytest.mark.asyncio
    async def test_return_all_scores(self, mock_module, sample_dataset, mock_metric):
        """Test return_all_scores parameter."""
        # With return_all_scores=True
        evaluator1 = Evaluate(
            dataset=sample_dataset,
            metric=mock_metric,
            return_all_scores=True,
            display_progress=False,
        )
        result1 = await evaluator1(mock_module)
        assert len(result1.scores) == 3
        assert len(result1.examples) == 3

        # With return_all_scores=False
        evaluator2 = Evaluate(
            dataset=sample_dataset,
            metric=mock_metric,
            return_all_scores=False,
            display_progress=False,
        )
        result2 = await evaluator2(mock_module)
        assert len(result2.scores) == 0
        assert len(result2.examples) == 0

    @pytest.mark.asyncio
    async def test_metadata_inclusion(self, mock_module, sample_dataset, mock_metric):
        """Test metadata is properly included."""
        evaluator = Evaluate(
            dataset=sample_dataset, metric=mock_metric, num_threads=3, display_progress=False
        )

        result = await evaluator(mock_module)

        # Check metadata fields
        assert "num_threads" in result.metadata
        assert result.metadata["num_threads"] == 3
        assert "dataset_size" in result.metadata
        assert result.metadata["dataset_size"] == 3
        assert "metric_name" in result.metadata
        assert result.metadata["metric_name"] == "MockMetric"

        # Duration should be positive
        assert result.total_time > 0
        assert result.avg_time_per_example > 0

    def test_run_sync(self, mock_module, sample_dataset, mock_metric):
        """Test synchronous wrapper."""
        evaluator = Evaluate(dataset=sample_dataset, metric=mock_metric, display_progress=False)

        result = evaluator.run_sync(mock_module)

        assert isinstance(result, EvaluationResult)
        assert result.score == pytest.approx(2 / 3, 0.01)
