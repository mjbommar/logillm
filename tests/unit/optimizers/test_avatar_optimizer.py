"""Tests for AvatarOptimizer."""

from unittest.mock import patch

import pytest

from logillm.core.avatar import ActionOutput, Avatar
from logillm.core.signatures import BaseSignature, FieldSpec
from logillm.core.tools.base import Tool
from logillm.core.types import FieldType, Prediction, Usage
from logillm.optimizers.avatar_optimizer import (
    AvatarOptimizer,
    ComparatorSignature,
    EvalResult,
    FeedbackBasedInstructionSignature,
)


@pytest.fixture
def mock_metric():
    """Create a mock evaluation metric."""

    def metric(example: dict, prediction: Prediction) -> float:
        """Simple metric that returns random scores."""
        # Simple heuristic: check if answer contains numbers for math questions
        if "answer" in prediction.outputs:
            answer = prediction.outputs["answer"].lower()
            if "error" in answer:
                return 0.0
            elif any(char.isdigit() for char in answer):
                return 0.8
            else:
                return 0.3
        return 0.5

    return metric


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""

    def calculator(expression: str) -> str:
        """Simple calculator."""
        try:
            result = eval(expression)
            return str(result)
        except Exception:
            return "Error: Invalid expression"

    def search(query: str) -> str:
        """Mock search tool."""
        return f"Results for: {query}"

    calc_tool = Tool(func=calculator, name="Calculator", desc="Math calculator")
    search_tool = Tool(func=search, name="Search", desc="Search tool")

    return [calc_tool, search_tool]


@pytest.fixture
def simple_signature():
    """Create a simple signature for testing."""
    return BaseSignature(
        input_fields={
            "question": FieldSpec(
                name="question",
                field_type=FieldType.INPUT,
                python_type=str,
                description="Question to answer",
                required=True,
            ),
        },
        output_fields={
            "answer": FieldSpec(
                name="answer",
                field_type=FieldType.OUTPUT,
                python_type=str,
                description="Answer to the question",
                required=True,
            ),
        },
        instructions="Answer questions using available tools",
    )


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return [
        {
            "inputs": {"question": "What is 2 + 2?"},
            "outputs": {"answer": "4"},
        },
        {
            "inputs": {"question": "What is 5 * 3?"},
            "outputs": {"answer": "15"},
        },
        {
            "inputs": {"question": "What is the capital of France?"},
            "outputs": {"answer": "Paris"},
        },
        {
            "inputs": {"question": "What is 10 / 2?"},
            "outputs": {"answer": "5"},
        },
    ]


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_eval_result_creation(self):
        """Test creating an EvalResult."""
        example = {"inputs": {"question": "test"}, "outputs": {"answer": "test"}}
        actions = [ActionOutput("Calculator", "2+2", "4")]

        result = EvalResult(
            example=example,
            score=0.8,
            actions=actions,
        )

        assert result.example == example
        assert result.score == 0.8
        assert result.actions == actions


class TestComparatorSignature:
    """Test ComparatorSignature."""

    def test_comparator_signature_creation(self):
        """Test creating ComparatorSignature."""
        signature = ComparatorSignature()

        assert "instruction" in signature.input_fields
        assert "actions" in signature.input_fields
        assert "pos_input_with_metrics" in signature.input_fields
        assert "neg_input_with_metrics" in signature.input_fields
        assert "feedback" in signature.output_fields
        assert signature.instructions is not None


class TestFeedbackBasedInstructionSignature:
    """Test FeedbackBasedInstructionSignature."""

    def test_feedback_instruction_signature_creation(self):
        """Test creating FeedbackBasedInstructionSignature."""
        signature = FeedbackBasedInstructionSignature()

        assert "previous_instruction" in signature.input_fields
        assert "feedback" in signature.input_fields
        assert "new_instruction" in signature.output_fields
        assert signature.instructions is not None


class TestAvatarOptimizer:
    """Test AvatarOptimizer."""

    def test_optimizer_initialization(self, mock_metric):
        """Test AvatarOptimizer initialization."""
        optimizer = AvatarOptimizer(
            metric=mock_metric,
            max_iters=5,
            lower_bound=0.2,
            upper_bound=0.8,
        )

        assert optimizer.metric == mock_metric
        assert optimizer.max_iters == 5
        assert optimizer.lower_bound == 0.2
        assert optimizer.upper_bound == 0.8
        assert optimizer.optimize_for == "max"

    def test_optimizer_initialization_no_metric(self):
        """Test AvatarOptimizer requires metric."""
        with pytest.raises(ValueError, match="metric.*cannot be None"):
            AvatarOptimizer(metric=None)

    def test_process_example_success(self, mock_metric, simple_signature, mock_tools):
        """Test processing a single example successfully."""
        optimizer = AvatarOptimizer(metric=mock_metric)
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        example = {
            "inputs": {"question": "What is 2 + 2?"},
            "outputs": {"answer": "4"},
        }

        # Mock avatar to return a prediction
        with patch.object(avatar, "call_sync") as mock_call:
            mock_call.return_value = Prediction(
                outputs={"answer": "4"},
                usage=Usage(),
                success=True,
            )

            score = optimizer._process_example(avatar, example, return_outputs=False)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_process_example_with_outputs(self, mock_metric, simple_signature, mock_tools):
        """Test processing example with return_outputs=True."""
        optimizer = AvatarOptimizer(metric=mock_metric)
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        example = {
            "inputs": {"question": "What is 2 + 2?"},
            "outputs": {"answer": "4"},
        }

        with patch.object(avatar, "call_sync") as mock_call:
            mock_prediction = Prediction(
                outputs={"answer": "4"},
                usage=Usage(),
                success=True,
            )
            mock_call.return_value = mock_prediction

            result = optimizer._process_example(avatar, example, return_outputs=True)
            returned_example, prediction, score = result

            assert returned_example == example
            assert prediction == mock_prediction
            assert isinstance(score, float)

    def test_process_example_error(self, mock_metric, simple_signature, mock_tools):
        """Test processing example with error."""
        optimizer = AvatarOptimizer(metric=mock_metric)
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        example = {
            "inputs": {"question": "What is 2 + 2?"},
            "outputs": {"answer": "4"},
        }

        with patch.object(avatar, "call_sync") as mock_call:
            mock_call.side_effect = Exception("Test error")

            score = optimizer._process_example(avatar, example, return_outputs=False)
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_avatar(self, mock_metric, simple_signature, mock_tools, sample_dataset):
        """Test evaluating Avatar on dataset."""
        optimizer = AvatarOptimizer(metric=mock_metric, num_threads=2)
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        with patch.object(optimizer, "_process_example") as mock_process:
            mock_process.return_value = 0.7

            score = await optimizer._evaluate_avatar(avatar, sample_dataset, return_outputs=False)

            assert isinstance(score, float)
            assert score == 0.7
            assert mock_process.call_count == len(sample_dataset)

    @pytest.mark.asyncio
    async def test_evaluate_avatar_with_outputs(
        self, mock_metric, simple_signature, mock_tools, sample_dataset
    ):
        """Test evaluating Avatar with return_outputs=True."""
        optimizer = AvatarOptimizer(metric=mock_metric, num_threads=2)
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        mock_prediction = Prediction(outputs={"answer": "test"}, usage=Usage(), success=True)

        with patch.object(optimizer, "_process_example") as mock_process:
            mock_process.return_value = (sample_dataset[0], mock_prediction, 0.8)

            score, results = await optimizer._evaluate_avatar(
                avatar, sample_dataset, return_outputs=True
            )

            assert isinstance(score, float)
            assert len(results) == len(sample_dataset)
            assert all(len(result) == 3 for result in results)

    @pytest.mark.asyncio
    async def test_get_pos_neg_results(
        self, mock_metric, simple_signature, mock_tools, sample_dataset
    ):
        """Test separating positive and negative results."""
        optimizer = AvatarOptimizer(
            metric=mock_metric,
            lower_bound=0.3,
            upper_bound=0.7,
        )
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        # Mock evaluation to return mix of high/low scores
        mock_results = [
            (sample_dataset[0], Prediction(outputs={"answer": "4"}, success=True), 0.8),  # Positive
            (
                sample_dataset[1],
                Prediction(outputs={"answer": "15"}, success=True),
                0.9,
            ),  # Positive
            (
                sample_dataset[2],
                Prediction(outputs={"answer": "error"}, success=True),
                0.1,
            ),  # Negative
            (
                sample_dataset[3],
                Prediction(outputs={"answer": "wrong"}, success=True),
                0.2,
            ),  # Negative
        ]

        with patch.object(optimizer, "_evaluate_avatar") as mock_eval:
            mock_eval.return_value = (0.5, mock_results)

            avg_score, pos_inputs, neg_inputs = await optimizer._get_pos_neg_results(
                avatar, sample_dataset
            )

            assert avg_score == 0.5
            assert len(pos_inputs) == 2  # Scores >= 0.7
            assert len(neg_inputs) == 2  # Scores <= 0.3
            assert all(isinstance(result, EvalResult) for result in pos_inputs)
            assert all(isinstance(result, EvalResult) for result in neg_inputs)

    @pytest.mark.asyncio
    async def test_get_pos_neg_results_no_positives(
        self, mock_metric, simple_signature, mock_tools, sample_dataset
    ):
        """Test error when no positive examples found."""
        optimizer = AvatarOptimizer(
            metric=mock_metric,
            lower_bound=0.3,
            upper_bound=0.9,  # Very high threshold
        )
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        # Mock evaluation to return only low scores
        mock_results = [
            (sample_dataset[0], Prediction(outputs={"answer": "error"}, success=True), 0.2),
            (sample_dataset[1], Prediction(outputs={"answer": "error"}, success=True), 0.1),
        ]

        with patch.object(optimizer, "_evaluate_avatar") as mock_eval:
            mock_eval.return_value = (0.15, mock_results)

            with pytest.raises(ValueError, match="No positive examples found"):
                await optimizer._get_pos_neg_results(avatar, sample_dataset[:2])

    @pytest.mark.asyncio
    async def test_get_pos_neg_results_no_negatives(
        self, mock_metric, simple_signature, mock_tools, sample_dataset
    ):
        """Test error when no negative examples found."""
        optimizer = AvatarOptimizer(
            metric=mock_metric,
            lower_bound=0.1,  # Very low threshold
            upper_bound=0.7,
        )
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        # Mock evaluation to return only high scores
        mock_results = [
            (sample_dataset[0], Prediction(outputs={"answer": "4"}, success=True), 0.8),
            (sample_dataset[1], Prediction(outputs={"answer": "15"}, success=True), 0.9),
        ]

        with patch.object(optimizer, "_evaluate_avatar") as mock_eval:
            mock_eval.return_value = (0.85, mock_results)

            with pytest.raises(ValueError, match="No negative examples found"):
                await optimizer._get_pos_neg_results(avatar, sample_dataset[:2])

    @pytest.mark.asyncio
    async def test_optimize_invalid_module(self, mock_metric, sample_dataset):
        """Test optimize with non-Avatar module."""
        optimizer = AvatarOptimizer(metric=mock_metric)

        class FakeModule:
            pass

        with pytest.raises(ValueError, match="only works with Avatar modules"):
            await optimizer.optimize(FakeModule(), sample_dataset)

    @pytest.mark.asyncio
    async def test_optimize_success(
        self, mock_metric, simple_signature, mock_tools, sample_dataset
    ):
        """Test successful optimization."""
        optimizer = AvatarOptimizer(
            metric=mock_metric,
            max_iters=2,
            lower_bound=0.3,
            upper_bound=0.7,
        )
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        # Mock the evaluation and feedback generation
        pos_result = EvalResult(
            example=sample_dataset[0],
            score=0.8,
            actions=[ActionOutput("Calculator", "2+2", "4")],
        )
        neg_result = EvalResult(
            example=sample_dataset[2],
            score=0.2,
            actions=[ActionOutput("Search", "Paris", "Error")],
        )

        with patch.object(optimizer, "_get_pos_neg_results") as mock_get_results:
            mock_get_results.return_value = (0.6, [pos_result], [neg_result])

            with patch.object(optimizer.comparator, "__call__") as mock_comparator:
                mock_comparator.return_value = Prediction(
                    outputs={"feedback": "Use calculator for math problems"},
                    success=True,
                )

                with patch.object(optimizer.feedback_instruction, "__call__") as mock_feedback:
                    mock_feedback.return_value = Prediction(
                        outputs={"new_instruction": "Improved instruction for better tool usage"},
                        success=True,
                    )

                    optimized_avatar = await optimizer.optimize(avatar, sample_dataset)

                    assert isinstance(optimized_avatar, Avatar)
                    assert optimized_avatar is not avatar  # Should be a copy

    @pytest.mark.asyncio
    async def test_optimize_with_sampling(
        self, mock_metric, simple_signature, mock_tools, sample_dataset
    ):
        """Test optimization with example sampling."""
        optimizer = AvatarOptimizer(
            metric=mock_metric,
            max_iters=1,
            max_positive_inputs=1,
            max_negative_inputs=1,
            lower_bound=0.3,
            upper_bound=0.7,
        )
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        # Create more examples than the sampling limits
        pos_results = [
            EvalResult(example=sample_dataset[0], score=0.8, actions=[]),
            EvalResult(example=sample_dataset[1], score=0.9, actions=[]),
        ]
        neg_results = [
            EvalResult(example=sample_dataset[2], score=0.2, actions=[]),
            EvalResult(example=sample_dataset[3], score=0.1, actions=[]),
        ]

        with patch.object(optimizer, "_get_pos_neg_results") as mock_get_results:
            mock_get_results.return_value = (0.5, pos_results, neg_results)

            with patch.object(optimizer.comparator, "__call__") as mock_comparator:
                mock_comparator.return_value = Prediction(
                    outputs={"feedback": "Test feedback"},
                    success=True,
                )

                with patch.object(optimizer.feedback_instruction, "__call__") as mock_feedback:
                    mock_feedback.return_value = Prediction(
                        outputs={"new_instruction": "Test instruction"},
                        success=True,
                    )

                    # Mock the sampling to verify it's called
                    with patch("logillm.optimizers.avatar_optimizer.sample") as mock_sample:
                        mock_sample.side_effect = lambda lst, n: lst[:n]  # Simple sampling

                        await optimizer.optimize(avatar, sample_dataset)

                        # Should sample both positive and negative examples
                        assert mock_sample.call_count == 2

    @pytest.mark.asyncio
    async def test_optimize_feedback_error(
        self, mock_metric, simple_signature, mock_tools, sample_dataset
    ):
        """Test optimization with feedback generation error."""
        optimizer = AvatarOptimizer(
            metric=mock_metric,
            max_iters=1,
            lower_bound=0.3,
            upper_bound=0.7,
        )
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        pos_result = EvalResult(example=sample_dataset[0], score=0.8, actions=[])
        neg_result = EvalResult(example=sample_dataset[1], score=0.2, actions=[])

        with patch.object(optimizer, "_get_pos_neg_results") as mock_get_results:
            mock_get_results.return_value = (0.5, [pos_result], [neg_result])

            with patch.object(optimizer.comparator, "__call__") as mock_comparator:
                mock_comparator.side_effect = Exception("Feedback error")

                with patch.object(optimizer.feedback_instruction, "__call__") as mock_feedback:
                    mock_feedback.return_value = Prediction(
                        outputs={"new_instruction": "Fallback instruction"},
                        success=True,
                    )

                    # Should handle error gracefully
                    optimized_avatar = await optimizer.optimize(avatar, sample_dataset)
                    assert isinstance(optimized_avatar, Avatar)

    @pytest.mark.asyncio
    async def test_optimize_instruction_error(
        self, mock_metric, simple_signature, mock_tools, sample_dataset
    ):
        """Test optimization with instruction generation error."""
        optimizer = AvatarOptimizer(
            metric=mock_metric,
            max_iters=1,
            lower_bound=0.3,
            upper_bound=0.7,
        )
        avatar = Avatar(signature=simple_signature, tools=mock_tools)

        pos_result = EvalResult(example=sample_dataset[0], score=0.8, actions=[])
        neg_result = EvalResult(example=sample_dataset[1], score=0.2, actions=[])

        with patch.object(optimizer, "_get_pos_neg_results") as mock_get_results:
            mock_get_results.return_value = (0.5, [pos_result], [neg_result])

            with patch.object(optimizer.comparator, "__call__") as mock_comparator:
                mock_comparator.return_value = Prediction(
                    outputs={"feedback": "Test feedback"},
                    success=True,
                )

                with patch.object(optimizer.feedback_instruction, "__call__") as mock_feedback:
                    mock_feedback.side_effect = Exception("Instruction error")

                    # Should handle error gracefully and keep original instruction
                    optimized_avatar = await optimizer.optimize(avatar, sample_dataset)
                    assert isinstance(optimized_avatar, Avatar)
