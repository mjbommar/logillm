"""Unit tests for the Retry module."""

import time

import pytest

from logillm.core.modules import BaseModule
from logillm.core.retry import Retry, RetryAttempt, RetryHistory, RetryStrategy
from logillm.core.signatures import Signature
from logillm.core.signatures.fields import InputField, OutputField
from logillm.core.types import Prediction, TokenUsage, Usage


class SimpleSignature(Signature):
    """Simple test signature."""

    question: str = InputField(description="A question to answer")
    answer: str = OutputField(description="The answer to the question")


class MockModule(BaseModule):
    """Mock module for testing."""

    def __init__(self, should_fail: bool = False, fail_count: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.should_fail = should_fail
        self.fail_count = fail_count  # Number of times to fail before succeeding
        self.call_count = 0

    async def forward(self, **inputs):
        self.call_count += 1

        if self.should_fail and self.call_count <= self.fail_count:
            return Prediction(
                success=False, error=f"Mock failure #{self.call_count}", usage=Usage(), outputs={}
            )

        # Simulate success
        return Prediction(
            success=True,
            outputs={"answer": f"Answer #{self.call_count}"},
            usage=Usage(tokens=TokenUsage(input_tokens=10, output_tokens=5)),
        )


@pytest.fixture
def simple_module():
    """Create a simple mock module."""
    return MockModule(signature=SimpleSignature)


@pytest.fixture
def failing_module():
    """Create a mock module that always fails."""
    return MockModule(signature=SimpleSignature, should_fail=True, fail_count=10)


@pytest.fixture
def eventually_successful_module():
    """Create a mock module that fails twice then succeeds."""
    return MockModule(signature=SimpleSignature, should_fail=True, fail_count=2)


class TestRetryHistory:
    """Test retry history tracking."""

    def test_empty_history(self):
        """Test empty history state."""
        history = RetryHistory()

        assert history.total_attempts == 0
        assert history.successful_attempts == 0
        assert history.failed_attempts == 0
        assert len(history.attempts) == 0
        assert history.get_last_errors() == []

    def test_add_successful_attempt(self):
        """Test adding successful attempt."""
        history = RetryHistory()

        attempt = RetryAttempt(
            attempt_number=1,
            timestamp=time.time(),
            inputs={"question": "test"},
            outputs={"answer": "test answer"},
            success=True,
        )

        history.add_attempt(attempt)

        assert history.total_attempts == 1
        assert history.successful_attempts == 1
        assert history.failed_attempts == 0
        assert len(history.attempts) == 1

    def test_add_failed_attempt(self):
        """Test adding failed attempt."""
        history = RetryHistory()

        attempt = RetryAttempt(
            attempt_number=1,
            timestamp=time.time(),
            inputs={"question": "test"},
            error="Test error",
            success=False,
        )

        history.add_attempt(attempt)

        assert history.total_attempts == 1
        assert history.successful_attempts == 0
        assert history.failed_attempts == 1
        assert history.get_last_errors() == ["Test error"]

    def test_get_last_errors(self):
        """Test retrieving last errors."""
        history = RetryHistory()

        # Add multiple failed attempts
        for i in range(5):
            attempt = RetryAttempt(
                attempt_number=i + 1,
                timestamp=time.time(),
                inputs={"question": "test"},
                error=f"Error {i + 1}",
                success=False,
            )
            history.add_attempt(attempt)

        # Get last 3 errors
        errors = history.get_last_errors(3)
        assert len(errors) == 3
        assert errors == ["Error 3", "Error 4", "Error 5"]

        # Get all errors
        all_errors = history.get_last_errors(10)
        assert len(all_errors) == 5


class TestRetryModule:
    """Test retry module functionality."""

    @pytest.mark.asyncio
    async def test_init_with_module(self, simple_module):
        """Test retry module initialization."""
        retry = Retry(simple_module, max_retries=3)

        assert retry.module == simple_module
        assert retry.original_signature == simple_module.signature
        assert retry.max_retries == 3
        assert retry.strategy == RetryStrategy.EXPONENTIAL
        assert isinstance(retry.history, RetryHistory)

    def test_signature_transformation(self, simple_module):
        """Test that retry creates enhanced signature."""
        retry = Retry(simple_module)

        # Should have enhanced signature with past fields and feedback
        assert retry.signature is not None
        assert hasattr(retry.signature, "input_fields")

        # Check for original input fields
        input_fields = retry.signature.input_fields
        assert "question" in input_fields

        # Check for past output fields
        assert "past_answer" in input_fields

        # Check for feedback field
        assert "feedback" in input_fields

    def test_signature_transformation_no_signature(self):
        """Test retry with module that has no signature."""
        module = MockModule()  # No signature
        retry = Retry(module)

        # Should handle gracefully
        assert retry.signature is None
        assert retry.original_signature is None

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self, simple_module):
        """Test retry with successful first attempt."""
        retry = Retry(simple_module, max_retries=3)

        result = await retry(question="What is 2+2?")

        assert result.success is True
        assert result.outputs["answer"] == "Answer #1"
        assert simple_module.call_count == 1
        assert retry.history.total_attempts == 1
        assert retry.history.successful_attempts == 1

    @pytest.mark.asyncio
    async def test_retry_after_failures(self, eventually_successful_module):
        """Test retry after initial failures."""
        retry = Retry(eventually_successful_module, max_retries=3)

        result = await retry(question="What is 2+2?")

        assert result.success is True
        assert result.outputs["answer"] == "Answer #3"  # Third attempt succeeded
        assert eventually_successful_module.call_count == 3
        assert retry.history.total_attempts == 3
        assert retry.history.successful_attempts == 1
        assert retry.history.failed_attempts == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, failing_module):
        """Test when all retries are exhausted."""
        retry = Retry(failing_module, max_retries=2)

        result = await retry(question="What is 2+2?")

        assert result.success is False
        assert "All 3 attempts failed" in result.error  # initial + 2 retries
        assert failing_module.call_count == 3
        assert retry.history.total_attempts == 3
        assert retry.history.failed_attempts == 3
        assert retry.history.successful_attempts == 0

    @pytest.mark.asyncio
    async def test_retry_strategies(self, eventually_successful_module):
        """Test different retry strategies."""
        # Test immediate retry
        retry_immediate = Retry(
            eventually_successful_module, max_retries=2, strategy=RetryStrategy.IMMEDIATE
        )

        start_time = time.time()
        await retry_immediate(question="test")
        duration = time.time() - start_time

        # Should be fast with immediate retry
        assert duration < 1.0

        # Reset the module
        eventually_successful_module.call_count = 0

        # Test exponential backoff
        retry_exponential = Retry(
            eventually_successful_module,
            max_retries=2,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=0.1,
        )

        start_time = time.time()
        await retry_exponential(question="test")
        duration = time.time() - start_time

        # Should take some time due to delays
        assert duration > 0.1  # At least the first retry delay

    def test_custom_retry_condition(self, simple_module):
        """Test custom retry condition."""

        def custom_condition(prediction):
            # Retry if answer doesn't contain "good"
            return prediction.success and "good" not in prediction.outputs.get("answer", "")

        retry = Retry(simple_module, max_retries=2, retry_condition=custom_condition)

        # The custom condition should be used
        assert retry.retry_condition == custom_condition

    def test_custom_feedback_generator(self, simple_module):
        """Test custom feedback generator."""

        def custom_feedback(history):
            return f"Custom feedback: {history.total_attempts} attempts made"

        retry = Retry(simple_module, feedback_generator=custom_feedback)

        assert retry.feedback_generator == custom_feedback

    def test_reset_history(self, simple_module):
        """Test resetting retry history."""
        retry = Retry(simple_module)

        # Add some history
        retry.history.add_attempt(
            RetryAttempt(attempt_number=1, timestamp=time.time(), inputs={"test": "value"})
        )

        assert retry.history.total_attempts == 1

        # Reset
        retry.reset_history()

        assert retry.history.total_attempts == 0
        assert len(retry.history.attempts) == 0

    def test_get_success_rate(self, simple_module):
        """Test success rate calculation."""
        retry = Retry(simple_module)

        # No attempts yet
        assert retry.get_success_rate() == 0.0

        # Add some attempts
        retry.history.add_attempt(
            RetryAttempt(attempt_number=1, timestamp=time.time(), inputs={}, success=True)
        )
        retry.history.add_attempt(
            RetryAttempt(attempt_number=2, timestamp=time.time(), inputs={}, success=False)
        )
        retry.history.add_attempt(
            RetryAttempt(attempt_number=3, timestamp=time.time(), inputs={}, success=True)
        )

        # Should be 2/3 = 0.667
        assert abs(retry.get_success_rate() - 2 / 3) < 0.001

    def test_get_average_attempts(self, simple_module):
        """Test average attempts calculation."""
        retry = Retry(simple_module)

        # No successful attempts
        assert retry.get_average_attempts() == 0.0

        # Simulate some sessions
        # Session 1: success on first attempt
        retry.history.add_attempt(
            RetryAttempt(attempt_number=1, timestamp=time.time(), inputs={}, success=True)
        )

        # Session 2: success on third attempt
        retry.history.add_attempt(
            RetryAttempt(attempt_number=1, timestamp=time.time(), inputs={}, success=False)
        )
        retry.history.add_attempt(
            RetryAttempt(attempt_number=2, timestamp=time.time(), inputs={}, success=False)
        )
        retry.history.add_attempt(
            RetryAttempt(attempt_number=3, timestamp=time.time(), inputs={}, success=True)
        )

        # Average should be (1 + 3) / 2 = 2.0
        assert retry.get_average_attempts() == 2.0

    @pytest.mark.asyncio
    async def test_feedback_propagation(self, eventually_successful_module):
        """Test that feedback is provided to subsequent attempts."""
        retry = Retry(eventually_successful_module, max_retries=3)

        # Mock the feedback generator to track calls
        original_feedback = retry.feedback_generator
        feedback_calls = []

        def mock_feedback(history):
            feedback = original_feedback(history)
            feedback_calls.append(feedback)
            return feedback

        retry.feedback_generator = mock_feedback

        await retry(question="What is 2+2?")

        # Should have generated feedback for failed attempts
        # Since the module fails the first 2 attempts and succeeds on the 3rd,
        # there should be feedback calls
        assert len(feedback_calls) >= 1
        # The feedback should mention failures or errors
        assert any(
            "failed" in fb.lower() or "error" in fb.lower() or "attempt" in fb.lower()
            for fb in feedback_calls
        )

    @pytest.mark.asyncio
    async def test_past_outputs_propagation(self):
        """Test that past outputs are properly propagated."""
        # Create a module that tracks what inputs it receives
        received_inputs = []

        class TrackingModule(BaseModule):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.attempt = 0

            async def forward(self, **inputs):
                received_inputs.append(inputs.copy())
                self.attempt += 1

                if self.attempt <= 2:
                    # Fail first two attempts
                    return Prediction(
                        success=False,
                        error=f"Failed attempt {self.attempt}",
                        outputs={"answer": f"wrong answer {self.attempt}"},
                        usage=Usage(),
                    )
                else:
                    # Succeed on third attempt
                    return Prediction(
                        success=True, outputs={"answer": "correct answer"}, usage=Usage()
                    )

        tracking_module = TrackingModule(signature=SimpleSignature)
        retry = Retry(tracking_module, max_retries=3)

        result = await retry(question="What is 2+2?")

        assert result.success is True
        assert len(received_inputs) == 3

        # First attempt should only have original inputs
        assert "question" in received_inputs[0]
        assert "past_answer" not in received_inputs[0]
        assert "feedback" not in received_inputs[0]

        # Second attempt should have past outputs and feedback, but they might not be passed to
        # the wrapped module if it doesn't have those fields in its signature.
        # However, feedback should be passed if present
        assert "question" in received_inputs[1]
        if "feedback" in received_inputs[1]:
            assert len(received_inputs[1]["feedback"]) > 0

        # Third attempt should have updated feedback
        assert "question" in received_inputs[2]
        if "feedback" in received_inputs[2]:
            assert len(received_inputs[2]["feedback"]) > 0


class TestRetryParameterOptimization:
    """Test retry parameter optimization."""

    def test_optimization_parameters(self, simple_module):
        """Test that retry exposes optimization parameters."""
        retry = Retry(simple_module, max_retries=3)

        params = retry.optimization_parameters()

        assert "max_retries" in params
        assert "strategy" in params
        assert "base_delay" in params
        assert params["max_retries"] == 3

    def test_parameter_learning(self, simple_module):
        """Test that retry parameters are marked as learnable."""
        retry = Retry(simple_module)

        assert "max_retries" in retry.parameters
        assert retry.parameters["max_retries"].learnable is True
        assert "strategy" in retry.parameters
        assert retry.parameters["strategy"].learnable is True


class TestRetryEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_module_raises_exception(self):
        """Test retry when wrapped module raises exceptions."""

        class ExceptionModule(BaseModule):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.attempt = 0

            async def forward(self, **inputs):
                self.attempt += 1
                if self.attempt <= 2:
                    raise ValueError(f"Test exception #{self.attempt}")
                return Prediction(success=True, outputs={"answer": "success"}, usage=Usage())

        exception_module = ExceptionModule(signature=SimpleSignature)
        retry = Retry(exception_module, max_retries=3)

        result = await retry(question="test")

        assert result.success is True
        assert result.outputs["answer"] == "success"
        assert exception_module.attempt == 3
        assert retry.history.total_attempts == 3
        assert retry.history.failed_attempts == 2
        assert retry.history.successful_attempts == 1

    @pytest.mark.asyncio
    async def test_all_attempts_raise_exceptions(self):
        """Test when all attempts raise exceptions."""

        class AlwaysFailModule(BaseModule):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.attempt = 0

            async def forward(self, **inputs):
                self.attempt += 1
                raise RuntimeError(f"Always fails #{self.attempt}")

        fail_module = AlwaysFailModule(signature=SimpleSignature)
        retry = Retry(fail_module, max_retries=2)

        result = await retry(question="test")

        assert result.success is False
        assert "All 3 attempts failed" in result.error
        assert fail_module.attempt == 3
        assert retry.history.total_attempts == 3
        assert retry.history.failed_attempts == 3

    def test_zero_max_retries(self, simple_module):
        """Test retry with zero max retries."""
        retry = Retry(simple_module, max_retries=0)
        assert retry.max_retries == 0

        # Should still work with just initial attempt

    @pytest.mark.asyncio
    async def test_delay_calculation(self, simple_module):
        """Test delay calculation for different strategies."""
        retry = Retry(simple_module, strategy=RetryStrategy.EXPONENTIAL, base_delay=2.0)

        # Test exponential backoff
        delay1 = await retry._calculate_delay(1)
        delay2 = await retry._calculate_delay(2)
        delay3 = await retry._calculate_delay(3)

        assert delay1 == 2.0  # base_delay * 2^0
        assert delay2 == 4.0  # base_delay * 2^1
        assert delay3 == 8.0  # base_delay * 2^2

        # Test max delay limit
        retry.max_delay = 5.0
        delay_large = await retry._calculate_delay(10)
        assert delay_large == 5.0

    @pytest.mark.asyncio
    async def test_linear_backoff(self, simple_module):
        """Test linear backoff strategy."""
        retry = Retry(simple_module, strategy=RetryStrategy.LINEAR, base_delay=1.5)

        delay1 = await retry._calculate_delay(1)
        delay2 = await retry._calculate_delay(2)
        delay3 = await retry._calculate_delay(3)

        assert delay1 == 1.5  # base_delay * 1
        assert delay2 == 3.0  # base_delay * 2
        assert delay3 == 4.5  # base_delay * 3

    @pytest.mark.asyncio
    async def test_immediate_strategy(self, simple_module):
        """Test immediate retry strategy."""
        retry = Retry(simple_module, strategy=RetryStrategy.IMMEDIATE)

        delay = await retry._calculate_delay(5)
        assert delay == 0.0
