"""Integration tests for OpenAI provider.

These tests use the real OpenAI API. Set OPENAI_API_KEY environment variable
to run these tests. They are marked with @pytest.mark.integration so they
can be skipped in CI.
"""

import os

import pytest
from pydantic import BaseModel

from logillm.core.predict import Predict
from logillm.providers import create_provider
from logillm.providers.openai import OpenAIProvider

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


class TestOpenAIProvider:
    """Test OpenAI provider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider with gpt-4.1-mini."""
        return create_provider(
            "openai",
            model="gpt-4.1-mini",  # Using cheaper mini model for testing
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_basic_completion(self, provider):
        """Test basic text completion."""
        messages = [{"role": "user", "content": "Say 'Hello, LogiLLM!' exactly."}]

        completion = await provider.complete(messages)

        assert completion.text
        assert "Hello, LogiLLM!" in completion.text
        assert completion.usage.tokens.input_tokens > 0
        assert completion.usage.tokens.output_tokens > 0
        assert completion.provider == "openai"
        assert completion.model  # Should be set

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_with_parameters(self, provider):
        """Test completion with various parameters."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 3."},
        ]

        completion = await provider.complete(
            messages,
            temperature=0.0,  # Deterministic
            max_tokens=20,
            stop=["\n"],
        )

        assert completion.text
        assert "1" in completion.text
        assert completion.usage.tokens.output_tokens <= 20

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_streaming(self, provider):
        """Test streaming completion."""
        messages = [{"role": "user", "content": "Count from 1 to 5 slowly."}]

        tokens = []
        async for token in provider.stream(messages, max_tokens=30):
            tokens.append(token)

        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert "1" in full_text

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_structured_output(self, provider):
        """Test structured output with Pydantic model."""

        class MathAnswer(BaseModel):
            """Math problem answer."""

            reasoning: str
            answer: int

        messages = [{"role": "user", "content": "What is 25 + 17? Think step by step."}]

        result = await provider.create_structured_completion(
            messages, response_format=MathAnswer, temperature=0.0
        )

        assert isinstance(result, MathAnswer)
        assert result.answer == 42
        assert result.reasoning  # Should have reasoning

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_error_handling(self, provider):
        """Test error handling for invalid requests."""
        messages = [{"role": "user", "content": "Test"}]

        # Test with invalid parameter
        with pytest.raises(Exception):  # Should raise validation error
            await provider.complete(
                messages,
                temperature=5.0,  # Out of range
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_retry_logic(self):
        """Test retry logic on failures."""
        # Create provider with low timeout to trigger retry
        provider = OpenAIProvider(
            model="gpt-4.1-mini",
            timeout=0.001,  # Very low timeout
            max_retries=2,
        )

        messages = [{"role": "user", "content": "Hello"}]

        # Should retry and eventually fail
        from logillm.providers.base import ProviderError

        with pytest.raises(ProviderError) as exc_info:
            await provider.complete_with_retry(messages)

        assert "Failed after" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_embeddings(self, provider):
        """Test embedding generation."""
        texts = ["Hello, world!", "How are you today?", "LogiLLM is great!"]

        embeddings = await provider.embed(texts)

        assert len(embeddings) == 3
        assert all(isinstance(e, list) for e in embeddings)
        assert all(len(e) > 0 for e in embeddings)
        assert all(isinstance(e[0], float) for e in embeddings)


class TestOpenAIWithPredict:
    """Test OpenAI provider with Predict module."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider."""
        return OpenAIProvider(model="gpt-4.1-mini")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_predict_with_signature(self, provider):
        """Test Predict module with OpenAI provider."""
        from logillm.providers import register_provider

        # Register the provider
        register_provider(provider, set_default=True)

        # Create Predict module
        predict = Predict("question -> answer")

        # Execute
        result = await predict.forward(
            question="What is the capital of France? Answer in one word."
        )

        assert result.success
        assert result.outputs["answer"]
        assert "Paris" in result.outputs["answer"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_chain_of_thought(self, provider):
        """Test chain-of-thought reasoning."""
        from logillm.core.predict import ChainOfThought
        from logillm.providers import register_provider

        register_provider(provider, set_default=True)

        # Create ChainOfThought module
        cot = ChainOfThought("question -> answer")

        # Execute
        result = await cot.forward(question="What is 17 * 3? Show your work.")

        assert result.success
        assert result.outputs.get("reasoning")
        assert result.outputs.get("answer")
        assert "51" in str(result.outputs["answer"])


class TestOpenAIOptimization:
    """Test optimization with OpenAI provider."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider."""
        return OpenAIProvider(model="gpt-4.1-mini")

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return [
            {"inputs": {"x": 2, "y": 3}, "outputs": {"result": 5}},
            {"inputs": {"x": 5, "y": 7}, "outputs": {"result": 12}},
            {"inputs": {"x": 10, "y": 15}, "outputs": {"result": 25}},
        ]

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_hyperparameter_optimization(self, provider, dataset):
        """Test hyperparameter optimization with real LLM."""
        from logillm.core.predict import Predict
        from logillm.optimizers.hyperparameter import HyperparameterOptimizer
        from logillm.providers import register_provider

        register_provider(provider, set_default=True)

        # Create module
        module = Predict("x: int, y: int -> result: int")

        # Define metric
        def accuracy_metric(pred_outputs, true_outputs):
            pred = pred_outputs.get("result", 0)
            true = true_outputs.get("result", 0)
            try:
                return 1.0 if int(pred) == int(true) else 0.0
            except:
                return 0.0

        # Create optimizer
        optimizer = HyperparameterOptimizer(
            metric=accuracy_metric,
            strategy="random",
            n_trials=3,  # Small number for testing
        )

        # Optimize
        result = await optimizer.optimize(module=module, trainset=dataset[:2], valset=dataset[2:])

        assert result.optimized_module
        assert result.best_score >= 0.0
        assert "best_config" in result.metadata

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_format_optimization(self, provider):
        """Test format optimization with real responses."""
        from logillm.core.predict import Predict
        from logillm.optimizers.format_optimizer import FormatOptimizer
        from logillm.providers import register_provider

        register_provider(provider, set_default=True)

        # Create module
        module = Predict("question -> answer")

        # Small dataset
        dataset = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What color is the sky?"}, "outputs": {"answer": "blue"}},
        ]

        # Define metric
        def exact_match(pred_outputs, true_outputs):
            pred = str(pred_outputs.get("answer", "")).lower().strip()
            true = str(true_outputs.get("answer", "")).lower().strip()
            return 1.0 if pred == true else 0.0

        # Create optimizer - test only JSON format for quick test
        from logillm.optimizers.format_optimizer import FormatOptimizerConfig, PromptFormat
        
        config = FormatOptimizerConfig(
            formats_to_test=[PromptFormat.JSON],  # Only test JSON format
            min_samples_per_format=1,  # Minimal samples for quick test
            max_samples_per_format=1,
        )
        optimizer = FormatOptimizer(metric=exact_match, config=config, track_by_model=True)

        # Optimize formats
        result = await optimizer.optimize(module=module, dataset=dataset)

        assert result.optimized_module
        assert result.metadata.get("best_format")
        assert result.metadata.get("format_scores")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
