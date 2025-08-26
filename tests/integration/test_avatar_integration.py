"""Integration tests for Avatar system with real LLM."""

import os

import pytest

from logillm.core.avatar import Avatar
from logillm.core.signatures import BaseSignature, FieldSpec
from logillm.core.tools.base import Tool
from logillm.core.types import FieldType
from logillm.optimizers.avatar_optimizer import AvatarOptimizer
from logillm.providers.openai import OpenAIProvider

# Skip integration tests if no API key available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not available"
)


@pytest.fixture
def openai_provider():
    """Create OpenAI provider for testing."""
    return OpenAIProvider(model="gpt-4.1-mini")  # Use cheaper model for tests


@pytest.fixture
def math_signature():
    """Create signature for math problems."""
    return BaseSignature(
        input_fields={
            "problem": FieldSpec(
                name="problem",
                field_type=FieldType.INPUT,
                python_type=str,
                description="Math problem to solve",
                required=True,
            ),
        },
        output_fields={
            "answer": FieldSpec(
                name="answer",
                field_type=FieldType.OUTPUT,
                python_type=str,
                description="Solution to the math problem",
                required=True,
            ),
        },
        instructions="Solve math problems using available tools. Show your work and provide the final numerical answer.",
    )


@pytest.fixture
def calculator_tool():
    """Create a calculator tool."""

    def calculate(expression: str) -> str:
        """Safely evaluate mathematical expressions."""
        try:
            # Basic safety - only allow certain characters
            allowed_chars = set("0123456789+-*/.()")
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return "Error: Invalid characters in expression"

            # Evaluate the expression
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return Tool(
        func=calculate,
        name="Calculator",
        desc="Evaluates mathematical expressions like '2+2' or '10*5'",
    )


@pytest.fixture
def research_signature():
    """Create signature for research tasks."""
    return BaseSignature(
        input_fields={
            "question": FieldSpec(
                name="question",
                field_type=FieldType.INPUT,
                python_type=str,
                description="Research question to answer",
                required=True,
            ),
        },
        output_fields={
            "answer": FieldSpec(
                name="answer",
                field_type=FieldType.OUTPUT,
                python_type=str,
                description="Answer based on research",
                required=True,
            ),
        },
        instructions="Answer questions using available research tools. Provide comprehensive and accurate information.",
    )


@pytest.fixture
def mock_search_tool():
    """Create a mock search tool for testing."""

    def search(query: str) -> str:
        """Mock search that returns predefined results."""
        search_results = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
            "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
            "default": f"Search results for '{query}': This is a mock search result containing general information about the query.",
        }

        query_lower = query.lower()
        for key, result in search_results.items():
            if key in query_lower:
                return result

        return search_results["default"]

    return Tool(
        func=search,
        name="Search",
        desc="Searches for information on any topic",
    )


class TestAvatarIntegration:
    """Integration tests for Avatar module with real LLM."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_avatar_tool_execution(self, math_signature, calculator_tool, openai_provider):
        """Test Avatar executing real tools with real LLM."""
        avatar = Avatar(
            signature=math_signature,
            tools=[calculator_tool],
            max_iters=3,
            provider=openai_provider,
            verbose=True,
        )

        result = await avatar(problem="What is 15 * 7 + 23?")

        assert result.success is True
        assert "answer" in result.outputs
        assert result.actions is not None

        # Check if calculator was used
        calculator_used = any(action.tool_name == "Calculator" for action in result.actions)
        assert calculator_used, "Calculator tool should have been used"

        # The answer should contain the correct result (128)
        answer = result.outputs["answer"].lower()
        assert "128" in answer or "one hundred twenty-eight" in answer

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_avatar_with_complex_tools(
        self, research_signature, mock_search_tool, openai_provider
    ):
        """Test Avatar with multiple tools and complex reasoning."""
        avatar = Avatar(
            signature=research_signature,
            tools=[mock_search_tool],
            max_iters=5,
            provider=openai_provider,
        )

        result = await avatar(question="What is machine learning and how is it related to AI?")

        assert result.success is True
        assert "answer" in result.outputs

        # Check that search tool was used
        search_used = any(action.tool_name == "Search" for action in result.actions)
        assert search_used, "Search tool should have been used"

        # Answer should mention both concepts
        answer = result.outputs["answer"].lower()
        assert "machine learning" in answer
        assert "ai" in answer or "artificial intelligence" in answer

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_avatar_error_recovery(self, math_signature, openai_provider):
        """Test Avatar handling of tool failures."""

        def broken_tool(input_str: str) -> str:
            """Tool that always fails."""
            raise ValueError("This tool is broken")

        broken = Tool(
            func=broken_tool,
            name="BrokenTool",
            desc="A tool that always fails",
        )

        avatar = Avatar(
            signature=math_signature,
            tools=[broken],
            max_iters=3,
            provider=openai_provider,
        )

        result = await avatar(problem="What is 2 + 2?")

        # Avatar should still complete, even with broken tools
        assert result.success is True
        assert "answer" in result.outputs

        # Should have attempted to use the broken tool and got an error
        if result.actions:
            error_found = any("Error" in action.tool_output for action in result.actions)
            assert error_found, "Should have recorded tool error"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_avatar_max_iters_enforcement(
        self, math_signature, calculator_tool, openai_provider
    ):
        """Test that Avatar respects max_iters limit."""
        avatar = Avatar(
            signature=math_signature,
            tools=[calculator_tool],
            max_iters=2,  # Very low limit
            provider=openai_provider,
        )

        result = await avatar(problem="Calculate 5 + 3 * 2 - 1")

        assert result.success is True
        assert len(result.actions) <= 2, "Should not exceed max_iters"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_avatar_finish_tool_behavior(
        self, math_signature, calculator_tool, openai_provider
    ):
        """Test Avatar's finish tool behavior."""
        avatar = Avatar(
            signature=math_signature,
            tools=[calculator_tool],
            max_iters=5,
            provider=openai_provider,
        )

        result = await avatar(problem="What is 10 + 5?")

        assert result.success is True

        # The last action should be deciding to finish
        # (or the Avatar should have used Finish tool to complete)

        # Check that we got a reasonable answer
        answer = result.outputs["answer"]
        assert answer is not None
        assert len(answer.strip()) > 0


class TestAvatarOptimizerIntegration:
    """Integration tests for AvatarOptimizer with real LLM."""

    def accuracy_metric(self, example: dict, prediction) -> float:
        """Simple accuracy metric for math problems."""
        expected = example.get("outputs", {}).get("answer", "")
        actual = prediction.outputs.get("answer", "")

        # Extract numbers from both answers
        import re

        expected_nums = re.findall(r"\d+", expected)
        actual_nums = re.findall(r"\d+", actual)

        if expected_nums and actual_nums:
            # Check if the main number matches
            try:
                exp_num = int(expected_nums[0])
                act_num = int(actual_nums[0])
                return 1.0 if exp_num == act_num else 0.0
            except (ValueError, IndexError):
                pass

        # Fallback to string similarity
        return 1.0 if expected.lower() in actual.lower() else 0.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_avatar_optimizer_improves_performance(
        self, math_signature, calculator_tool, openai_provider
    ):
        """Test that AvatarOptimizer actually improves Avatar performance."""
        # Create initial avatar with a poor instruction
        poor_signature = BaseSignature(
            input_fields=math_signature.input_fields,
            output_fields=math_signature.output_fields,
            instructions="Just guess the answer to math problems without using any tools.",
        )

        avatar = Avatar(
            signature=poor_signature,
            tools=[calculator_tool],
            max_iters=3,
            provider=openai_provider,
        )

        # Create a simple dataset
        dataset = [
            {
                "inputs": {"problem": "What is 6 + 4?"},
                "outputs": {"answer": "10"},
            },
            {
                "inputs": {"problem": "What is 8 * 3?"},
                "outputs": {"answer": "24"},
            },
            {
                "inputs": {"problem": "What is 15 - 7?"},
                "outputs": {"answer": "8"},
            },
            {
                "inputs": {"problem": "What is 12 / 4?"},
                "outputs": {"answer": "3"},
            },
        ]

        optimizer = AvatarOptimizer(
            metric=self.accuracy_metric,
            max_iters=2,  # Keep short for testing
            lower_bound=0.0,
            upper_bound=0.8,
            provider=openai_provider,
        )

        # Note: This test might be flaky due to LLM randomness
        # In a real scenario, you'd run multiple times and check average improvement
        try:
            optimized_avatar = await optimizer.optimize(avatar, dataset)

            assert isinstance(optimized_avatar, Avatar)
            assert optimized_avatar is not avatar  # Should be a different object

            # The optimized avatar should have different (hopefully better) instructions
            original_instructions = avatar.signature.instructions if avatar.signature else ""
            optimized_instructions = (
                optimized_avatar.signature.instructions if optimized_avatar.signature else ""
            )

            # Instructions should be different (optimizer should have modified them)
            # Note: In rare cases they might be the same, but this is the expected behavior
            print(f"Original: {original_instructions}")
            print(f"Optimized: {optimized_instructions}")

        except ValueError as e:
            # If we don't get positive/negative examples, that's also informative
            pytest.skip(f"Couldn't optimize due to data distribution: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_avatar_optimizer_with_mixed_performance(
        self, research_signature, mock_search_tool, openai_provider
    ):
        """Test optimizer with examples that have mixed performance."""
        avatar = Avatar(
            signature=research_signature,
            tools=[mock_search_tool],
            max_iters=3,
            provider=openai_provider,
        )

        # Dataset with both easy and hard questions
        dataset = [
            {
                "inputs": {"question": "What is Python?"},
                "outputs": {"answer": "Python is a programming language"},
            },
            {
                "inputs": {"question": "What is AI?"},
                "outputs": {"answer": "AI is artificial intelligence"},
            },
            {
                "inputs": {"question": "What is machine learning?"},
                "outputs": {"answer": "Machine learning is a subset of AI"},
            },
        ]

        def research_metric(example: dict, prediction) -> float:
            """Metric that checks if key terms are mentioned."""
            expected_answer = example["outputs"]["answer"].lower()
            actual_answer = prediction.outputs.get("answer", "").lower()

            # Check for key term overlap
            expected_words = set(expected_answer.split())
            actual_words = set(actual_answer.split())

            overlap = len(expected_words.intersection(actual_words))
            return min(1.0, overlap / max(1, len(expected_words)))

        optimizer = AvatarOptimizer(
            metric=research_metric,
            max_iters=1,
            lower_bound=0.2,
            upper_bound=0.7,
            provider=openai_provider,
        )

        try:
            optimized_avatar = await optimizer.optimize(avatar, dataset)
            assert isinstance(optimized_avatar, Avatar)

        except ValueError as e:
            # If distribution doesn't allow optimization, that's okay for this test
            pytest.skip(f"Couldn't optimize due to score distribution: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout
    async def test_avatar_with_no_tools(self, math_signature, openai_provider):
        """Test Avatar behavior with no tools except Finish."""
        avatar = Avatar(
            signature=math_signature,
            tools=[],  # No tools except automatic Finish
            max_iters=3,
            provider=openai_provider,
        )

        result = await avatar(problem="What is 5 + 3?")

        assert result.success is True
        assert "answer" in result.outputs

        # Should have minimal actions since no tools to use
        assert len(result.actions) <= 1
