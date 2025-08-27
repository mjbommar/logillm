"""Mock components for testing optimization."""

from typing import Any

from logillm.core.modules import Module
from logillm.core.parameters import ParamDomain, ParamSpec, ParamType
from logillm.core.providers import Provider
from logillm.core.types import Configuration, Prediction, Usage


class MockModule(Module):
    """Mock module for testing optimization."""

    def __init__(self, behavior="linear", seed=None, **kwargs):
        """Initialize mock module.

        Args:
            behavior: How the module responds to parameters
                - "linear": Score increases linearly with temperature
                - "quadratic": Score peaks at temperature=0.5
                - "random": Random scores
                - "failing": Randomly fails
            seed: Random seed for reproducible behavior
        """
        super().__init__(**kwargs)
        self.behavior = behavior
        self.call_count = 0
        self.provider = MockProvider()
        self.seed = seed
        import random

        # Always use a Random instance (not the module) for pickleability
        self.rng = random.Random(seed) if seed is not None else random.Random()

    async def forward(self, **inputs: Any) -> Prediction:
        """Mock forward pass."""
        self.call_count += 1

        # Simulate different behaviors based on config
        temperature = self.config.get("temperature", 0.7)
        top_p = self.config.get("top_p", 0.9)

        if self.behavior == "linear":
            # Higher temperature = better score
            score = temperature
        elif self.behavior == "quadratic":
            # Peak at temperature=0.5
            score = 1.0 - abs(temperature - 0.5) * 2
        elif self.behavior == "random":
            score = self.rng.random()
        elif self.behavior == "failing":
            if self.rng.random() < 0.3:  # 30% failure rate
                raise RuntimeError("Mock failure")
            score = 0.5
        else:
            score = 0.5

        # Factor in top_p slightly
        score = score * 0.8 + top_p * 0.2

        return Prediction(
            outputs={"score": score, "result": f"temp={temperature:.2f}"},
            usage=Usage(tokens={"prompt": 10, "completion": 5, "total": 15}),
            success=True,
            metadata={"temperature": temperature, "top_p": top_p},
        )

    def reset_stats(self):
        """Reset call statistics."""
        self.call_count = 0


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(self):
        """Initialize mock provider."""
        super().__init__(provider_name="mock", model="mock-model")
        self.name = "mock"
        self.model = "mock-model"

    async def _complete_impl(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        """Mock completion implementation."""
        from logillm.core.types import Completion
        # Simple mock response
        if messages:
            last_message = messages[-1].get("content", "")
            return Completion(
                text=f"Mock response to: {last_message}",
                usage=None,
                metadata={},
                finish_reason="stop",
                model=self.model,
                provider=self.name
            )
        return Completion(
            text="Mock response",
            usage=None,
            metadata={},
            finish_reason="stop",
            model=self.model,
            provider=self.name
        )

    async def complete(self, prompt: str, **kwargs) -> str:
        """Mock completion for backward compatibility."""
        messages = [{"role": "user", "content": prompt}]
        result = await self._complete_impl(messages, **kwargs)
        return result.text if hasattr(result, "text") else str(result)

    async def chat(self, messages: list, **kwargs) -> str:
        """Mock chat."""
        result = await self._complete_impl(messages, **kwargs)
        return result.text if hasattr(result, "text") else "Mock chat response"

    async def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Mock embedding."""
        # Return mock embedding vectors for each text
        return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]

    def get_param_specs(self) -> dict[str, ParamSpec]:
        """Get parameter specifications."""
        return {
            "temperature": ParamSpec(
                name="temperature",
                param_type=ParamType.FLOAT,
                domain=ParamDomain.GENERATION,
                description="Mock temperature",
                default=0.7,
                range=(0.0, 2.0),
            ),
            "top_p": ParamSpec(
                name="top_p",
                param_type=ParamType.FLOAT,
                domain=ParamDomain.GENERATION,
                description="Mock top-p",
                default=0.9,
                range=(0.0, 1.0),
            ),
            "mock_categorical": ParamSpec(
                name="mock_categorical",
                param_type=ParamType.CATEGORICAL,
                domain=ParamDomain.GENERATION,
                description="Mock categorical param",
                default="option_a",
                choices=["option_a", "option_b", "option_c"],
            ),
        }


class MockMetric:
    """Mock evaluation metric."""

    def __init__(self, target_value=0.8):
        """Initialize mock metric.

        Args:
            target_value: Target value for optimization
        """
        self.target_value = target_value
        self.call_count = 0
        self.name = "mock_metric"

    def __call__(self, prediction: Any, target: Any) -> float:
        """Evaluate prediction against target."""
        self.call_count += 1

        # Extract score from prediction outputs
        if isinstance(prediction, dict):
            score = prediction.get("score", 0.5)
        elif hasattr(prediction, "outputs"):
            score = prediction.outputs.get("score", 0.5)
        else:
            score = 0.5

        # Compare to target
        if isinstance(target, dict):
            target_score = target.get("score", self.target_value)
        else:
            target_score = self.target_value

        # Return similarity (1.0 - distance)
        return 1.0 - abs(score - target_score)

    def reset_stats(self):
        """Reset call statistics."""
        self.call_count = 0


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, size=10, task_type="general"):
        """Initialize mock dataset.

        Args:
            size: Number of examples
            task_type: Type of task (affects example structure)
        """
        self.size = size
        self.task_type = task_type
        self.examples = self._generate_examples()

    def _generate_examples(self) -> list[dict[str, Any]]:
        """Generate mock examples."""
        examples = []

        for i in range(self.size):
            if self.task_type == "classification":
                example = {
                    "inputs": {"text": f"Sample text {i}", "label": i % 3},
                    "outputs": {"score": 0.7 + (i % 3) * 0.1},
                }
            elif self.task_type == "generation":
                example = {
                    "inputs": {"prompt": f"Generate text {i}"},
                    "outputs": {"text": f"Generated text {i}", "score": 0.8},
                }
            elif self.task_type == "qa":
                example = {
                    "inputs": {"question": f"Question {i}?", "context": f"Context {i}"},
                    "outputs": {"answer": f"Answer {i}", "score": 0.75},
                }
            else:  # general
                example = {
                    "inputs": {"input": f"Input {i}"},
                    "outputs": {"output": f"Output {i}", "score": 0.7},
                }

            examples.append(example)

        return examples

    def get_train_val_split(self, val_ratio=0.2):
        """Split into train and validation sets."""
        val_size = int(self.size * val_ratio)
        return self.examples[val_size:], self.examples[:val_size]


class OptimizationMonitor:
    """Monitor optimization progress for testing."""

    def __init__(self):
        """Initialize monitor."""
        self.history = []
        self.best_score = float("-inf")
        self.best_config = None
        self.iterations = 0

    def record(self, config: Configuration, score: float):
        """Record optimization step."""
        self.iterations += 1
        self.history.append({"iteration": self.iterations, "config": config.copy(), "score": score})

        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy()

    def get_improvement(self) -> float:
        """Get improvement from first to best."""
        if not self.history:
            return 0.0
        first_score = self.history[0]["score"]
        return self.best_score - first_score

    def is_converged(self, patience=5, threshold=0.001) -> bool:
        """Check if optimization has converged."""
        if len(self.history) < patience:
            return False

        recent_scores = [h["score"] for h in self.history[-patience:]]
        score_variance = max(recent_scores) - min(recent_scores)
        return score_variance < threshold

    def reset(self):
        """Reset monitor."""
        self.history = []
        self.best_score = float("-inf")
        self.best_config = None
        self.iterations = 0
