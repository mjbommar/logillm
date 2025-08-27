"""Unit tests for SIMBA optimizer."""

import asyncio
import random
from unittest.mock import AsyncMock, Mock, patch

import pytest

from logillm.core.modules import Module, Parameter
from logillm.core.optimizers import AccuracyMetric
from logillm.core.types import Prediction, Usage
from logillm.exceptions import OptimizationError
from logillm.optimizers.simba import SIMBA, SIMBAConfig
from logillm.optimizers.simba_utils import (
    MockLM,
    OfferFeedback,
    inspect_modules,
    prepare_models_for_resampling,
    recursive_mask,
    wrap_program,
)


class MockModule(Module):
    """Mock module for testing."""

    def __init__(self, success_rate: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.success_rate = success_rate
        self.call_count = 0
        self.simba_idx = 0

    async def forward(self, **inputs):
        self.call_count += 1

        # Simulate success/failure based on success_rate
        success = random.random() < self.success_rate

        if success:
            outputs = {"answer": f"response_to_{inputs.get('question', 'unknown')}"}
        else:
            outputs = {"answer": "error"}

        return Prediction(outputs=outputs, usage=Usage(), success=success)


@pytest.fixture
def mock_metric():
    """Create a mock accuracy metric."""
    return AccuracyMetric("answer")


@pytest.fixture
def mock_module():
    """Create a mock module."""
    return MockModule(success_rate=0.8)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    return [
        {"inputs": {"question": f"What is {i} + {i}?"}, "outputs": {"answer": str(2 * i)}}
        for i in range(100)  # Large enough for SIMBA's default batch size
    ]


@pytest.fixture
def simba_optimizer(mock_metric):
    """Create SIMBA optimizer with small parameters for testing."""
    return SIMBA(
        metric=mock_metric,
        bsize=5,
        num_candidates=2,
        max_steps=2,
        max_demos=2,
        num_threads=2,
    )


class TestSIMBAConfig:
    """Test SIMBA configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SIMBAConfig()
        assert config.bsize == 32
        assert config.num_candidates == 6
        assert config.max_steps == 8
        assert config.max_demos == 4
        assert config.demo_input_field_maxlen == 100_000
        assert config.temperature_for_sampling == 0.2
        assert config.temperature_for_candidates == 0.2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SIMBAConfig(
            bsize=16,
            num_candidates=3,
            max_steps=5,
            max_demos=2,
        )
        assert config.bsize == 16
        assert config.num_candidates == 3
        assert config.max_steps == 5
        assert config.max_demos == 2


class TestSIMBAOptimizer:
    """Test SIMBA optimizer functionality."""

    def test_initialization(self, mock_metric):
        """Test SIMBA initialization."""
        optimizer = SIMBA(
            metric=mock_metric,
            bsize=10,
            num_candidates=3,
            max_steps=5,
        )

        assert optimizer.metric == mock_metric
        assert optimizer.bsize == 10
        assert optimizer.num_candidates == 3
        assert optimizer.max_steps == 5
        assert len(optimizer.strategies) == 2  # append_demo and append_rule

    def test_initialization_no_demos(self, mock_metric):
        """Test SIMBA initialization with max_demos=0."""
        optimizer = SIMBA(
            metric=mock_metric,
            max_demos=0,
        )

        assert len(optimizer.strategies) == 1  # only append_rule

    def test_dataset_too_small(self, simba_optimizer, mock_module):
        """Test error when dataset is too small."""
        small_dataset = [{"inputs": {"x": 1}, "outputs": {"y": 2}}]  # Only 1 example

        with pytest.raises(OptimizationError, match="Dataset too small"):
            asyncio.run(simba_optimizer.optimize(mock_module, small_dataset))

    @pytest.mark.asyncio
    async def test_softmax_sampling(self, simba_optimizer):
        """Test softmax sampling functionality."""
        rng = random.Random(42)
        program_idxs = [0, 1, 2]

        def score_fn(idx):
            return [0.1, 0.5, 0.9][idx]  # Different scores for each program

        # Test normal sampling
        result = simba_optimizer._softmax_sample(rng, program_idxs, 0.5, score_fn)
        assert result in program_idxs

        # Test with empty program list
        with pytest.raises(ValueError, match="No programs available"):
            simba_optimizer._softmax_sample(rng, [], 0.5, score_fn)

    def test_named_predictors(self, simba_optimizer, mock_module):
        """Test extracting named predictors from module."""
        # Test with module that has predictor parameters
        mock_module.parameters["test_predictor"] = Parameter(value="test", learnable=True)

        predictors = simba_optimizer._named_predictors(mock_module)
        assert len(predictors) == 1
        assert predictors[0][0] == "test_predictor"

        # Test with module that has no predictor parameters
        mock_module.parameters.clear()
        predictors = simba_optimizer._named_predictors(mock_module)
        assert len(predictors) == 1
        assert predictors[0][0] == "MockModule"

    def test_drop_random_demos(self, simba_optimizer, mock_module):
        """Test dropping random demonstrations."""
        rng = random.Random(42)

        # Add some demo parameters
        demos = [{"inputs": {"x": i}, "outputs": {"y": i * 2}} for i in range(10)]
        mock_module.parameters["demonstrations"] = Parameter(value=demos, learnable=True)

        # Mock named_predictors to return our module
        with patch.object(simba_optimizer, "_named_predictors") as mock_named:
            mock_named.return_value = [("test", mock_module)]

            original_count = len(demos)
            simba_optimizer._drop_random_demos(mock_module, rng)

            # Should have dropped some demos if max_demos constraint is active
            remaining = len(mock_module.parameters["demonstrations"].value)
            if simba_optimizer.max_demos > 0 and original_count > simba_optimizer.max_demos:
                assert remaining < original_count

    @pytest.mark.asyncio
    async def test_append_demo(self, simba_optimizer, mock_module):
        """Test appending demonstration strategy."""
        # Create mock bucket with trace
        mock_predictor = Mock()
        mock_predictor_id = id(mock_predictor)

        bucket = [
            {
                "score": 0.9,
                "trace": [(mock_predictor, {"question": "test"}, {"answer": "test_answer"})],
                "example": {"inputs": {"question": "test"}, "outputs": {"answer": "test_answer"}},
            }
        ]

        predictor2name = {mock_predictor_id: "test_predictor"}

        # Mock named_predictors
        with patch.object(simba_optimizer, "_named_predictors") as mock_named:
            mock_named.return_value = [("test_predictor", mock_module)]

            result = await simba_optimizer._append_demo(
                bucket, mock_module, predictor2name=predictor2name
            )

            assert result is True
            assert "demonstrations" in mock_module.parameters
            assert len(mock_module.parameters["demonstrations"].value) == 1

    @pytest.mark.asyncio
    async def test_append_rule(self, simba_optimizer, mock_module):
        """Test appending rule strategy."""
        # Create mock bucket with good and bad examples
        bucket = [
            {
                "score": 0.9,
                "trace": [],
                "example": {"inputs": {"question": "test"}, "outputs": {"answer": "correct"}},
                "prediction": {"answer": "correct"},
            },
            {
                "score": 0.1,
                "trace": [],
                "example": {"inputs": {"question": "test"}, "outputs": {"answer": "correct"}},
                "prediction": {"answer": "wrong"},
            },
        ]

        # Mock the feedback module to return some advice
        mock_feedback_result = Prediction(
            outputs={"module_advice": '{"test_predictor": "Be more careful with answers."}'},
            success=True,
        )

        with patch("logillm.core.predict.Predict") as mock_predict_class:
            mock_predict_instance = AsyncMock()
            mock_predict_instance.return_value = mock_feedback_result
            mock_predict_class.return_value = mock_predict_instance

            with patch.object(simba_optimizer, "_named_predictors") as mock_named:
                mock_named.return_value = [("test_predictor", mock_module)]

                result = await simba_optimizer._append_rule(
                    bucket, mock_module, predictor2name={}, batch_10p_score=0.0, batch_90p_score=1.0
                )

                assert result is True
                assert "instruction" in mock_module.parameters
                assert "Be more careful" in mock_module.parameters["instruction"].value

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, simba_optimizer):
        """Test batch evaluation functionality."""

        # Create mock wrapped programs and examples
        def mock_wrapped_program(example):
            return {"prediction": {"answer": "test"}, "trace": [], "score": 0.8, "example": example}

        examples = [{"inputs": {"question": f"test{i}"}} for i in range(3)]
        exec_pairs = [(mock_wrapped_program, ex) for ex in examples]

        results = await simba_optimizer._evaluate_batch(exec_pairs)

        assert len(results) == 3
        for result in results:
            assert "prediction" in result
            assert "score" in result
            assert result["score"] == 0.8

    @pytest.mark.asyncio
    async def test_optimize_full_workflow(self, simba_optimizer, mock_module, mock_dataset):
        """Test the complete optimization workflow."""
        # Use a smaller dataset for testing
        small_dataset = mock_dataset[:10]

        # Mock various methods to avoid complex interactions
        with patch.object(simba_optimizer, "_evaluate_batch") as mock_eval:
            # Mock evaluation results
            mock_eval.return_value = [
                {
                    "prediction": {"answer": f"pred_{i}"},
                    "trace": [],
                    "score": 0.7 + (i % 3) * 0.1,  # Varying scores
                    "example": small_dataset[i % len(small_dataset)],
                }
                for i in range(20)  # Enough for batch processing
            ]

            with patch.object(simba_optimizer, "_append_demo") as mock_append_demo:
                mock_append_demo.return_value = True

                with patch.object(simba_optimizer, "_append_rule") as mock_append_rule:
                    mock_append_rule.return_value = True

                    result = await simba_optimizer.optimize(mock_module, small_dataset)

                    assert result is not None
                    assert result.optimized_module is not None
                    assert result.iterations == simba_optimizer.max_steps
                    assert result.best_score >= 0.0
                    assert "trial_logs" in result.metadata


class TestSIMBAUtils:
    """Test SIMBA utility functions."""

    @pytest.mark.asyncio
    async def test_prepare_models_for_resampling(self):
        """Test model preparation for resampling."""
        mock_module = MockModule()
        mock_module.metadata["temperature"] = 0.5

        models = await prepare_models_for_resampling(mock_module, 3)

        assert len(models) == 3
        assert all(isinstance(model, MockLM) for model in models)
        assert models[0].temperature == 0.5  # Base temperature

    def test_mock_lm(self):
        """Test MockLM functionality."""
        lm = MockLM(temperature=0.7)
        assert lm.temperature == 0.7
        assert lm.kwargs["temperature"] == 0.7

        copied_lm = lm.copy(temperature=0.9)
        assert copied_lm.temperature == 0.9
        assert lm.temperature == 0.7  # Original unchanged

    def test_wrap_program(self, mock_metric, mock_module):
        """Test program wrapping functionality."""
        wrapped = wrap_program(mock_module, mock_metric)

        example = {"inputs": {"question": "test"}, "outputs": {"answer": "test"}}
        result = wrapped(example)

        assert "prediction" in result
        assert "trace" in result
        assert "score" in result
        assert "example" in result
        assert result["example"] == example

    def test_inspect_modules(self, mock_module):
        """Test module inspection."""
        # Add some parameters and signature
        mock_module.parameters["test_param"] = Parameter(value="test", learnable=True)

        result = inspect_modules(mock_module)

        assert "Module MockModule" in result
        assert "Parameters:" in result
        assert "test_param" in result

    def test_recursive_mask(self):
        """Test recursive object masking."""
        # Test basic serializable objects
        assert recursive_mask({"a": 1, "b": "test"}) == {"a": 1, "b": "test"}
        assert recursive_mask([1, 2, "test"]) == [1, 2, "test"]

        # Test non-serializable objects
        class NonSerializable:
            def __init__(self):
                self.value = lambda x: x  # Lambda is not JSON serializable

        obj = {"data": NonSerializable()}
        result = recursive_mask(obj)
        # The result should be the __dict__ of the non-serializable object
        assert isinstance(result["data"], dict)
        assert "value" in result["data"]
        # The lambda inside should be converted to a string
        assert isinstance(result["data"]["value"], str)

        # Test nested structures
        nested = {
            "good": [1, 2, 3],
            "bad": {"func": lambda x: x},
            "mixed": [1, NonSerializable(), "text"],
        }

        result = recursive_mask(nested)
        assert result["good"] == [1, 2, 3]
        # The lambda function should be converted to a placeholder string
        assert isinstance(result["bad"]["func"], str)
        assert "non-serializable" in result["bad"]["func"]
        assert result["mixed"][0] == 1
        # The NonSerializable object should be converted to its dict representation
        assert isinstance(result["mixed"][1], dict)
        assert result["mixed"][2] == "text"

    def test_offer_feedback_signature(self):
        """Test OfferFeedback signature."""
        # OfferFeedback is already a signature instance
        signature = OfferFeedback

        # Check that the signature has the expected fields
        input_fields = signature.input_fields
        assert "program_code" in input_fields
        assert "better_program_trajectory" in input_fields
        assert "worse_program_trajectory" in input_fields
        assert "module_names" in input_fields

        # Check output fields exist
        output_fields = signature.output_fields
        assert "discussion" in output_fields
        assert "module_advice" in output_fields

        # Check instructions are present
        assert signature.instructions
        assert "trajectories" in signature.instructions
        assert "advice" in signature.instructions


# Integration-style tests (still using mocks but testing more complete workflows)


class TestSIMBAIntegration:
    """Integration tests for SIMBA optimizer."""

    @pytest.mark.asyncio
    async def test_temperature_variation_sampling(self, mock_metric):
        """Test that different temperature models produce varied sampling."""
        optimizer = SIMBA(
            metric=mock_metric,
            bsize=5,
            num_candidates=3,
            max_steps=1,
            temperature_for_sampling=0.1,  # Low temperature - more deterministic
        )

        MockModule()

        # Create programs with different scores
        programs = [MockModule() for _ in range(3)]
        for i, prog in enumerate(programs):
            prog.simba_idx = i

        program_scores = {0: [0.5], 1: [0.8], 2: [0.3]}

        def score_fn(idx):
            scores = program_scores.get(idx, [0])
            return sum(scores) / len(scores)

        # With low temperature, should favor higher-scoring programs
        rng = random.Random(42)
        samples = []
        for _ in range(10):
            sample = optimizer._softmax_sample(rng, [0, 1, 2], 0.1, score_fn)
            samples.append(sample)

        # Program 1 (score 0.8) should be sampled most often
        assert samples.count(1) > samples.count(0)
        assert samples.count(1) > samples.count(2)

    @pytest.mark.asyncio
    async def test_demo_management_workflow(self, mock_metric):
        """Test the full demo management workflow."""
        optimizer = SIMBA(
            metric=mock_metric,
            bsize=3,
            max_demos=2,  # Small limit for testing
        )

        mock_module = MockModule()

        # Add initial demos
        initial_demos = [
            {"inputs": {"q": f"test{i}"}, "outputs": {"a": f"answer{i}"}}
            for i in range(5)  # More than max_demos
        ]
        mock_module.parameters["demonstrations"] = Parameter(value=initial_demos, learnable=True)

        # Test demo dropping
        rng = random.Random(42)
        with patch.object(optimizer, "_named_predictors") as mock_named:
            mock_named.return_value = [("test", mock_module)]

            original_count = len(initial_demos)
            optimizer._drop_random_demos(mock_module, rng)

            remaining_count = len(mock_module.parameters["demonstrations"].value)
            assert remaining_count <= original_count

        # Test demo addition
        mock_predictor = Mock()
        bucket = [
            {
                "score": 0.9,
                "trace": [(mock_predictor, {"question": "new"}, {"answer": "new_answer"})],
            }
        ]

        with patch.object(optimizer, "_named_predictors") as mock_named:
            mock_named.return_value = [("test", mock_module)]

            success = await optimizer._append_demo(
                bucket, mock_module, predictor2name={id(mock_predictor): "test"}
            )
            assert success

    @pytest.mark.asyncio
    async def test_rule_generation_edge_cases(self, mock_metric):
        """Test rule generation with various edge cases."""
        optimizer = SIMBA(metric=mock_metric, bsize=2)
        mock_module = MockModule()

        # Test: Empty bucket
        result = await optimizer._append_rule([], mock_module, predictor2name={})
        assert result is False

        # Test: Single example bucket
        single_bucket = [
            {
                "score": 0.5,
                "trace": [],
                "example": {"inputs": {"q": "test"}},
                "prediction": {"answer": "test"},
            }
        ]
        result = await optimizer._append_rule(single_bucket, mock_module, predictor2name={})
        assert result is False

        # Test: Good score below threshold
        low_score_bucket = [
            {"score": 0.05, "trace": [], "example": {"inputs": {}}, "prediction": {}},
            {"score": 0.01, "trace": [], "example": {"inputs": {}}, "prediction": {}},
        ]
        result = await optimizer._append_rule(
            low_score_bucket,
            mock_module,
            predictor2name={},
            batch_10p_score=0.1,
            batch_90p_score=0.9,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_rule_generation_perfect_scores(self, mock_metric):
        """Test rule generation when all scores are perfect (edge case that caused N/A bug)."""
        optimizer = SIMBA(metric=mock_metric, bsize=2)
        mock_module = MockModule()

        # Test: All perfect scores (would trigger the N/A assignment)
        perfect_bucket = [
            {
                "score": 1.0,
                "trace": [],
                "example": {"inputs": {"q": "test1"}},
                "prediction": {"answer": "correct1"},
            },
            {
                "score": 1.0,
                "trace": [],
                "example": {"inputs": {"q": "test2"}},
                "prediction": {"answer": "correct2"},
            },
        ]

        # Mock the feedback module to return advice
        mock_feedback_result = Prediction(
            outputs={"module_advice": '{"test_predictor": "Keep up the good work!"}'},
            success=True,
        )

        with patch("logillm.core.predict.Predict") as mock_predict_class:
            mock_predict_instance = AsyncMock()
            mock_predict_instance.return_value = mock_feedback_result
            mock_predict_class.return_value = mock_predict_instance

            with patch.object(optimizer, "_named_predictors") as mock_named:
                mock_named.return_value = [("test_predictor", mock_module)]

                # This should not raise a "could not convert string to float: 'N/A'" error
                result = await optimizer._append_rule(
                    perfect_bucket,
                    mock_module,
                    predictor2name={},
                    batch_10p_score=0.9,  # High 10th percentile
                    batch_90p_score=1.0,  # Perfect 90th percentile
                )

                # Should handle the N/A case gracefully
                assert result is True or result is False  # Either outcome is acceptable
                # The key is that it doesn't raise an exception
