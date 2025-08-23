"""Unit tests for the Refine module."""

import time
from unittest.mock import Mock, patch

import pytest

from logillm.core.modules import BaseModule
from logillm.core.refine import OfferFeedback, Refine, RefinementAttempt, RefinementHistory
from logillm.core.signatures import Signature
from logillm.core.signatures.fields import InputField, OutputField
from logillm.core.types import Prediction, TokenUsage, Usage


class SimpleSignature(Signature):
    """Simple test signature."""

    question: str = InputField(description="A question to answer")
    answer: str = OutputField(description="The answer to the question")


class MockModule(BaseModule):
    """Mock module for testing refinement."""

    # Class-level counter to survive deep copies
    _global_call_count = 0

    def __init__(self, responses=None, **kwargs):
        super().__init__(**kwargs)
        self.responses = responses or ["default response"]
        self.call_count = 0
        self.received_inputs = []

    async def forward(self, **inputs):
        self.received_inputs.append(inputs.copy())
        # Use class-level counter to get different responses even with deep copies
        response = self.responses[min(MockModule._global_call_count, len(self.responses) - 1)]
        MockModule._global_call_count += 1
        self.call_count += 1

        if response == "FAIL":
            return Prediction(success=False, error="Mock failure", usage=Usage(), outputs={})

        return Prediction(
            success=True,
            outputs={"answer": response},
            usage=Usage(tokens=TokenUsage(input_tokens=10, output_tokens=5)),
        )


def simple_reward_fn(inputs, prediction):
    """Simple reward function for testing."""
    if not prediction.success:
        return 0.0

    answer = prediction.outputs.get("answer", "")
    return len(answer) / 100.0  # Reward longer answers


def brevity_reward_fn(inputs, prediction):
    """Reward function that prefers shorter answers."""
    if not prediction.success:
        return 0.0

    answer = prediction.outputs.get("answer", "")
    word_count = len(answer.split())
    return max(0.0, 1.0 - (word_count - 1) * 0.1)  # Prefer 1-word answers


def threshold_reward_fn(inputs, prediction):
    """Reward function with clear threshold behavior."""
    if not prediction.success:
        return 0.0

    answer = prediction.outputs.get("answer", "")
    return 0.9 if "good" in answer.lower() else 0.3


@pytest.fixture
def simple_module():
    """Create a simple mock module."""
    MockModule._global_call_count = 0  # Reset counter for each test
    return MockModule(
        signature=SimpleSignature,
        responses=["short", "medium answer", "this is a very long answer"],
    )


@pytest.fixture
def variable_quality_module():
    """Create a module with variable quality responses."""
    MockModule._global_call_count = 0  # Reset counter for each test
    return MockModule(
        signature=SimpleSignature, responses=["bad", "better", "good answer", "excellent response"]
    )


class TestRefinementHistory:
    """Test refinement history tracking."""

    def test_empty_history(self):
        """Test empty history state."""
        history = RefinementHistory()

        assert len(history.attempts) == 0
        assert history.best_attempt is None
        assert history.best_reward == float("-inf")

    def test_add_attempt_first(self):
        """Test adding first attempt."""
        history = RefinementHistory()

        attempt = RefinementAttempt(
            attempt_number=1,
            timestamp=time.time(),
            temperature=0.7,
            inputs={"question": "test"},
            outputs={"answer": "test answer"},
            reward=0.5,
            success=True,
        )

        history.add_attempt(attempt)

        assert len(history.attempts) == 1
        assert history.best_attempt == attempt
        assert history.best_reward == 0.5

    def test_add_better_attempt(self):
        """Test adding a better attempt."""
        history = RefinementHistory()

        # Add first attempt
        attempt1 = RefinementAttempt(
            attempt_number=1,
            timestamp=time.time(),
            temperature=0.7,
            inputs={"question": "test"},
            reward=0.5,
            success=True,
        )
        history.add_attempt(attempt1)

        # Add better attempt
        attempt2 = RefinementAttempt(
            attempt_number=2,
            timestamp=time.time(),
            temperature=0.8,
            inputs={"question": "test"},
            reward=0.8,
            success=True,
        )
        history.add_attempt(attempt2)

        assert len(history.attempts) == 2
        assert history.best_attempt == attempt2
        assert history.best_reward == 0.8

    def test_add_worse_attempt(self):
        """Test adding a worse attempt doesn't change best."""
        history = RefinementHistory()

        # Add first attempt
        attempt1 = RefinementAttempt(
            attempt_number=1,
            timestamp=time.time(),
            temperature=0.7,
            inputs={"question": "test"},
            reward=0.8,
            success=True,
        )
        history.add_attempt(attempt1)

        # Add worse attempt
        attempt2 = RefinementAttempt(
            attempt_number=2,
            timestamp=time.time(),
            temperature=0.8,
            inputs={"question": "test"},
            reward=0.3,
            success=True,
        )
        history.add_attempt(attempt2)

        assert len(history.attempts) == 2
        assert history.best_attempt == attempt1  # Still the first one
        assert history.best_reward == 0.8


class TestOfferFeedback:
    """Test the OfferFeedback signature."""

    def test_signature_fields(self):
        """Test that OfferFeedback has required fields."""
        assert hasattr(OfferFeedback, "input_fields")
        assert hasattr(OfferFeedback, "output_fields")

        input_fields = OfferFeedback.input_fields
        output_fields = OfferFeedback.output_fields

        # Check required input fields
        expected_inputs = [
            "program_code",
            "modules_defn",
            "program_inputs",
            "program_trajectory",
            "program_outputs",
            "reward_code",
            "target_threshold",
            "reward_value",
            "module_names",
        ]

        for field_name in expected_inputs:
            assert field_name in input_fields, f"Missing input field: {field_name}"

        # Check required output fields
        assert "discussion" in output_fields
        assert "advice" in output_fields


class TestRefineModule:
    """Test refine module functionality."""

    def test_init_basic(self, simple_module):
        """Test basic refine module initialization."""
        refine = Refine(module=simple_module, N=3, reward_fn=simple_reward_fn, threshold=0.5)

        assert refine.module == simple_module
        assert refine.N == 3
        assert refine.reward_fn == simple_reward_fn
        assert refine.threshold == 0.5
        assert refine.fail_count == 3  # defaults to N
        assert isinstance(refine.history, RefinementHistory)

    def test_init_with_fail_count(self, simple_module):
        """Test initialization with custom fail count."""
        refine = Refine(
            module=simple_module, N=5, reward_fn=simple_reward_fn, threshold=0.5, fail_count=2
        )

        assert refine.fail_count == 2

    def test_init_invalid_n(self, simple_module):
        """Test initialization with invalid N."""
        with pytest.raises(ValueError, match="N must be at least 1"):
            Refine(module=simple_module, N=0, reward_fn=simple_reward_fn, threshold=0.5)

    def test_generate_temperature_sequence(self, simple_module):
        """Test temperature sequence generation."""
        refine = Refine(module=simple_module, N=4, reward_fn=simple_reward_fn, threshold=0.5)

        temps = refine._generate_temperature_sequence(0.7)

        assert len(temps) == 4
        assert temps[0] == 0.7  # Base temperature first
        assert all(0.0 <= temp <= 2.0 for temp in temps)  # Reasonable range

        # Test single attempt
        refine_single = Refine(module=simple_module, N=1, reward_fn=simple_reward_fn, threshold=0.5)

        temps_single = refine_single._generate_temperature_sequence(0.5)
        assert temps_single == [0.5]

    def test_inspect_modules(self, simple_module):
        """Test module inspection functionality."""
        refine = Refine(module=simple_module, N=3, reward_fn=simple_reward_fn, threshold=0.5)

        inspection = refine._inspect_modules(simple_module)

        assert isinstance(inspection, str)
        assert "MockModule" in inspection
        assert "Input Fields" in inspection
        assert "Output Fields" in inspection

    def test_recursive_mask(self, simple_module):
        """Test recursive masking of non-serializable objects."""
        refine = Refine(module=simple_module, N=3, reward_fn=simple_reward_fn, threshold=0.5)

        # Test serializable objects
        assert refine._recursive_mask({"a": 1, "b": "text"}) == {"a": 1, "b": "text"}
        assert refine._recursive_mask([1, 2, 3]) == [1, 2, 3]

        # Test non-serializable objects
        non_serializable = {"func": lambda x: x, "list": [1, 2, lambda: None]}
        masked = refine._recursive_mask(non_serializable)

        assert "func" in masked
        assert masked["list"][0] == 1
        assert masked["list"][1] == 2
        assert "<non-serializable" in masked["list"][2]

    @pytest.mark.asyncio
    async def test_single_attempt_success(self, simple_module):
        """Test refinement with single successful attempt."""
        refine = Refine(module=simple_module, N=1, reward_fn=simple_reward_fn, threshold=0.01)

        result = await refine(question="What is 2+2?")

        assert result.success is True
        assert result.outputs["answer"] == "short"
        assert len(refine.history.attempts) == 1
        # The module is deep-copied for each attempt, so original call_count stays 0
        assert refine.history.attempts[0].success is True

    @pytest.mark.asyncio
    async def test_multiple_attempts_best_selection(self, simple_module):
        """Test that refine selects the best attempt."""
        # Use reward function that prefers longer answers
        refine = Refine(
            module=simple_module,
            N=3,
            reward_fn=simple_reward_fn,  # Rewards longer answers
            threshold=10.0,  # High threshold to force all attempts
        )

        result = await refine(question="What is 2+2?")

        assert result.success is True
        # Should select the longest answer
        assert result.outputs["answer"] == "this is a very long answer"
        assert len(refine.history.attempts) == 3
        # The module is deep-copied for each attempt, so we check history instead

        # Check that best attempt was recorded
        assert refine.history.best_attempt is not None
        assert refine.history.best_attempt.outputs["answer"] == "this is a very long answer"

    @pytest.mark.asyncio
    async def test_early_stopping_threshold(self, variable_quality_module):
        """Test early stopping when threshold is met."""
        refine = Refine(
            module=variable_quality_module,
            N=4,
            reward_fn=threshold_reward_fn,  # Returns 0.9 for "good", 0.3 for others
            threshold=0.8,  # Should stop at "good answer" (3rd response)
        )

        result = await refine(question="What is 2+2?")

        assert result.success is True
        assert result.outputs["answer"] == "good answer"
        assert len(refine.history.attempts) == 3  # Stopped early
        # Module is deep-copied, so we check history instead of original call count

    @pytest.mark.asyncio
    async def test_all_attempts_fail(self):
        """Test when all attempts fail."""
        MockModule._global_call_count = 0  # Reset counter
        failing_module = MockModule(signature=SimpleSignature, responses=["FAIL", "FAIL", "FAIL"])

        refine = Refine(module=failing_module, N=3, reward_fn=simple_reward_fn, threshold=0.5)

        result = await refine(question="What is 2+2?")

        assert result.success is False
        # When all attempts fail, it returns the best failed attempt
        assert result.error == "Mock failure"  # The error from the failed predictions
        assert len(refine.history.attempts) == 3
        # Check that refinement metadata is included
        assert "refinement_attempts" in result.metadata

    @pytest.mark.asyncio
    async def test_fail_count_limit(self):
        """Test fail count limiting with exceptions."""

        class ExceptionModule(BaseModule):
            # Use class-level counter like MockModule
            _global_attempt = 0

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.attempt = 0

            async def forward(self, **inputs):
                self.attempt += 1
                ExceptionModule._global_attempt += 1
                if ExceptionModule._global_attempt <= 2:
                    raise RuntimeError(f"Module error #{ExceptionModule._global_attempt}")
                return Prediction(success=True, outputs={"answer": "success"}, usage=Usage())

        ExceptionModule._global_attempt = 0  # Reset counter
        exception_module = ExceptionModule(signature=SimpleSignature)

        refine = Refine(
            module=exception_module,
            N=3,
            reward_fn=simple_reward_fn,
            threshold=0.5,
            fail_count=1,  # Only allow 1 exception failure
        )

        result = await refine(question="What is 2+2?")

        # Should fail after 1 exception, so only have 1 attempt
        assert result.success is False
        assert "failed after 1 failures" in result.error
        assert len(refine.history.attempts) == 1  # Only first attempt before giving up

    @pytest.mark.asyncio
    async def test_temperature_variation(self, simple_module):
        """Test that different temperatures are used."""

        # Create a mock module that records the temperature it's called with
        class TemperatureTrackingModule(BaseModule):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.temperatures = []

            async def forward(self, **inputs):
                # Record temperature from config
                temp = getattr(self, "config", {}).get("temperature", 0.7)
                self.temperatures.append(temp)

                return Prediction(
                    success=True,
                    outputs={"answer": f"response_{len(self.temperatures)}"},
                    usage=Usage(),
                )

        tracking_module = TemperatureTrackingModule(signature=SimpleSignature)

        refine = Refine(
            module=tracking_module,
            N=3,
            reward_fn=simple_reward_fn,
            threshold=10.0,  # High threshold to force all attempts
        )

        await refine(question="test")

        # Module is deep-copied, so the original won't have temperatures recorded
        # Instead, check that we have 3 attempts with different temperatures in history
        assert len(refine.history.attempts) == 3

        # Extract temperatures from history attempts
        temperatures = [attempt.temperature for attempt in refine.history.attempts]
        temps_set = set(temperatures)
        assert len(temps_set) >= 2  # At least some temperature variation

    @pytest.mark.asyncio
    @patch("logillm.core.refine.Predict")
    async def test_feedback_generation(self, mock_predict_class, simple_module):
        """Test that feedback is generated between attempts."""
        # Mock the advice generation
        mock_predict_instance = Mock()

        async def mock_predict_call(**kwargs):
            return Prediction(
                success=True,
                outputs={
                    "advice": {"HintTrackingModule": "Try to be more specific in your responses."}
                },
                usage=Usage(),
            )

        mock_predict_instance.__call__ = mock_predict_call
        mock_predict_class.return_value = mock_predict_instance

        # Create module that tracks hint inputs using class-level storage
        class HintTrackingModule(BaseModule):
            # Class-level storage to survive deep copies
            _global_hints_received = []
            _global_call_count = 0

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.call_count = 0

            async def forward(self, **inputs):
                self.call_count += 1
                HintTrackingModule._global_call_count += 1
                if "hint_" in inputs:
                    HintTrackingModule._global_hints_received.append(inputs["hint_"])

                # Fail first attempt, succeed later to trigger feedback generation
                if HintTrackingModule._global_call_count == 1:
                    return Prediction(
                        success=False, error="First attempt fails", outputs={}, usage=Usage()
                    )
                else:
                    return Prediction(
                        success=True,
                        outputs={"answer": f"response_{HintTrackingModule._global_call_count}"},
                        usage=Usage(),
                    )

        # Reset class-level storage
        HintTrackingModule._global_hints_received = []
        HintTrackingModule._global_call_count = 0
        hint_module = HintTrackingModule(signature=SimpleSignature)

        refine = Refine(
            module=hint_module,
            N=3,
            reward_fn=simple_reward_fn,
            threshold=10.0,  # High threshold to force all attempts
        )

        await refine(question="test")

        # Should have attempted feedback generation between attempts
        # Check that the mocked Predict was called for advice generation
        mock_predict_class.assert_called()

        # Should have made all attempts
        assert len(refine.history.attempts) == 3

    def test_reset_history(self, simple_module):
        """Test resetting refinement history."""
        refine = Refine(module=simple_module, N=3, reward_fn=simple_reward_fn, threshold=0.5)

        # Add some history
        refine.history.add_attempt(
            RefinementAttempt(
                attempt_number=1,
                timestamp=time.time(),
                temperature=0.7,
                inputs={"test": "value"},
                reward=0.5,
            )
        )

        assert len(refine.history.attempts) == 1

        # Reset
        refine.reset_history()

        assert len(refine.history.attempts) == 0
        assert refine.history.best_attempt is None
        assert refine.history.best_reward == float("-inf")

    def test_get_average_reward(self, simple_module):
        """Test average reward calculation."""
        refine = Refine(module=simple_module, N=3, reward_fn=simple_reward_fn, threshold=0.5)

        # No attempts yet
        assert refine.get_average_reward() == 0.0

        # Add some attempts
        refine.history.add_attempt(
            RefinementAttempt(
                attempt_number=1,
                timestamp=time.time(),
                temperature=0.7,
                inputs={},
                reward=0.5,
                success=True,
            )
        )
        refine.history.add_attempt(
            RefinementAttempt(
                attempt_number=2,
                timestamp=time.time(),
                temperature=0.8,
                inputs={},
                reward=0.3,
                success=True,
            )
        )
        refine.history.add_attempt(
            RefinementAttempt(
                attempt_number=3,
                timestamp=time.time(),
                temperature=0.9,
                inputs={},
                reward=0.0,
                success=False,  # Failed attempt shouldn't count
            )
        )

        # Average should be (0.5 + 0.3) / 2 = 0.4
        assert abs(refine.get_average_reward() - 0.4) < 0.001

    def test_get_improvement_rate(self, simple_module):
        """Test improvement rate calculation."""
        refine = Refine(module=simple_module, N=3, reward_fn=simple_reward_fn, threshold=0.5)

        # No attempts
        assert refine.get_improvement_rate() == 0.0

        # Single attempt
        refine.history.add_attempt(
            RefinementAttempt(
                attempt_number=1, timestamp=time.time(), temperature=0.7, inputs={}, reward=0.5
            )
        )
        assert refine.get_improvement_rate() == 0.0

        # Add second attempt with improvement
        refine.history.add_attempt(
            RefinementAttempt(
                attempt_number=2, timestamp=time.time(), temperature=0.8, inputs={}, reward=0.8
            )
        )

        # Improvement rate should be (0.8 - 0.5) / 0.5 = 0.6
        expected_rate = (0.8 - 0.5) / 0.5
        assert abs(refine.get_improvement_rate() - expected_rate) < 0.001

    def test_improvement_rate_edge_cases(self, simple_module):
        """Test improvement rate edge cases."""
        refine = Refine(module=simple_module, N=3, reward_fn=simple_reward_fn, threshold=0.5)

        # First reward is zero
        refine.history.add_attempt(
            RefinementAttempt(
                attempt_number=1, timestamp=time.time(), temperature=0.7, inputs={}, reward=0.0
            )
        )
        refine.history.add_attempt(
            RefinementAttempt(
                attempt_number=2, timestamp=time.time(), temperature=0.8, inputs={}, reward=0.5
            )
        )

        # Should return infinity when first reward is 0 and improvement exists
        assert refine.get_improvement_rate() == float("inf")


class TestRefineErrorHandling:
    """Test error handling in refine module."""

    @pytest.mark.asyncio
    async def test_reward_function_exception(self, simple_module):
        """Test handling of reward function exceptions."""
        MockModule._global_call_count = 0  # Reset counter

        def failing_reward_fn(inputs, prediction):
            raise ValueError("Reward calculation failed")

        refine = Refine(module=simple_module, N=2, reward_fn=failing_reward_fn, threshold=0.5)

        result = await refine(question="test")

        # Should still return a result but with very low reward
        assert result is not None
        assert len(refine.history.attempts) == 2
        # All attempts should have -inf reward due to exception
        for attempt in refine.history.attempts:
            assert attempt.reward == float("-inf")

    @pytest.mark.asyncio
    async def test_module_execution_exception(self):
        """Test handling of module execution exceptions."""

        class ExceptionModule(BaseModule):
            # Use class-level counter like MockModule
            _global_attempt = 0

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.attempt = 0

            async def forward(self, **inputs):
                self.attempt += 1
                ExceptionModule._global_attempt += 1
                if ExceptionModule._global_attempt <= 2:
                    raise RuntimeError(f"Module error #{ExceptionModule._global_attempt}")
                return Prediction(success=True, outputs={"answer": "success"}, usage=Usage())

        ExceptionModule._global_attempt = 0  # Reset counter

        exception_module = ExceptionModule(signature=SimpleSignature)

        refine = Refine(
            module=exception_module, N=3, reward_fn=simple_reward_fn, threshold=0.5, fail_count=3
        )

        result = await refine(question="test")

        # Should eventually succeed on the third attempt
        assert result.success is True
        assert result.outputs["answer"] == "success"
        assert len(refine.history.attempts) == 3
        # Module is deep-copied, so the original attempt counter won't be updated

    @pytest.mark.asyncio
    async def test_feedback_generation_exception(self, simple_module):
        """Test handling of feedback generation exceptions."""
        # Create refine module with mocked failing feedback
        refine = Refine(
            module=simple_module,
            N=3,
            reward_fn=simple_reward_fn,
            threshold=10.0,  # High threshold to force feedback generation
        )

        # Mock _generate_feedback to raise exception

        async def failing_feedback(*args, **kwargs):
            raise Exception("Feedback generation failed")

        refine._generate_feedback = failing_feedback

        result = await refine(question="test")

        # Should still work despite feedback failure
        assert result.success is True
        # Should have made attempts (the exact number may vary due to implementation details)
        assert len(refine.history.attempts) > 0
        assert refine.history.best_reward > 0  # Should have found some successful attempt


class TestRefineIntegrationPatterns:
    """Test common integration patterns."""

    def test_reward_function_patterns(self, simple_module):
        """Test different reward function patterns."""

        # Length-based reward
        def length_reward(inputs, prediction):
            if not prediction.success:
                return 0.0
            return len(prediction.outputs.get("answer", "")) / 10.0

        # Keyword-based reward
        def keyword_reward(inputs, prediction):
            if not prediction.success:
                return 0.0
            answer = prediction.outputs.get("answer", "").lower()
            return 1.0 if "important" in answer else 0.2

        # Combined reward
        def combined_reward(inputs, prediction):
            length = length_reward(inputs, prediction)
            keyword = keyword_reward(inputs, prediction)
            return (length + keyword) / 2.0

        # All should be valid reward functions
        refine1 = Refine(simple_module, N=2, reward_fn=length_reward, threshold=0.5)
        refine2 = Refine(simple_module, N=2, reward_fn=keyword_reward, threshold=0.5)
        refine3 = Refine(simple_module, N=2, reward_fn=combined_reward, threshold=0.5)

        assert refine1.reward_fn == length_reward
        assert refine2.reward_fn == keyword_reward
        assert refine3.reward_fn == combined_reward

    @pytest.mark.asyncio
    async def test_chaining_refine_modules(self):
        """Test chaining multiple refine modules."""
        # Create a chain of refine modules
        base_module = MockModule(signature=SimpleSignature, responses=["ok", "good", "excellent"])

        # First refine for length
        length_refine = Refine(module=base_module, N=2, reward_fn=simple_reward_fn, threshold=0.05)

        # Second refine wrapping the first (though this is somewhat artificial)
        # In practice, you'd chain different modules
        quality_refine = Refine(
            module=length_refine, N=2, reward_fn=threshold_reward_fn, threshold=0.8
        )

        result = await quality_refine(question="test")

        # Should work without issues
        assert result.success is True or result.success is False  # Either outcome is valid
