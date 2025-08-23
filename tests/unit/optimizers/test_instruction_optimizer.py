"""Tests for InstructionOptimizer."""

import pytest

from logillm.optimizers.instruction_optimizer import InstructionOptimizer
from tests.unit.fixtures.mock_components import MockMetric, MockModule


class TestInstructionOptimizer:
    """Test suite for InstructionOptimizer."""

    @pytest.fixture
    def classification_dataset(self):
        """Create a classification dataset."""
        return [
            {"inputs": {"text": "I love this movie!"}, "outputs": {"label": "positive"}},
            {"inputs": {"text": "This is terrible."}, "outputs": {"label": "negative"}},
            {"inputs": {"text": "Great product!"}, "outputs": {"label": "positive"}},
            {"inputs": {"text": "Waste of money."}, "outputs": {"label": "negative"}},
            {"inputs": {"text": "Amazing experience!"}, "outputs": {"label": "positive"}},
        ]

    @pytest.fixture
    def qa_dataset(self):
        """Create a QA dataset."""
        return [
            {
                "inputs": {"question": "What is 2+2?", "context": "Basic math"},
                "outputs": {"answer": "4"},
            },
            {
                "inputs": {"question": "Capital of France?", "context": "Geography"},
                "outputs": {"answer": "Paris"},
            },
            {
                "inputs": {"question": "Who wrote Hamlet?", "context": "Literature"},
                "outputs": {"answer": "Shakespeare"},
            },
        ]

    @pytest.fixture
    def transform_dataset(self):
        """Create a transformation dataset."""
        return [
            {"inputs": {"number": 5}, "outputs": {"doubled": 10}},
            {"inputs": {"number": 3}, "outputs": {"doubled": 6}},
            {"inputs": {"number": 7}, "outputs": {"doubled": 14}},
        ]

    @pytest.mark.asyncio
    async def test_basic_functionality(self, classification_dataset):
        """Test basic InstructionOptimizer functionality."""
        module = MockModule(behavior="linear")
        metric = MockMetric()
        optimizer = InstructionOptimizer(metric=metric, num_candidates=3, seed=42)

        result = await optimizer.optimize(module=module, dataset=classification_dataset)

        # Check that instruction was optimized
        assert "instruction" in result.optimized_module.parameters
        instruction = result.optimized_module.parameters["instruction"].value
        assert isinstance(instruction, str)
        assert len(instruction) > 0

        # Check metadata
        assert "best_instruction" in result.metadata
        assert "task_analysis" in result.metadata
        assert "selection_strategy" in result.metadata

    @pytest.mark.asyncio
    async def test_task_analysis(self, classification_dataset, qa_dataset, transform_dataset):
        """Test task analysis for different dataset types."""
        optimizer = InstructionOptimizer(metric=MockMetric(), num_candidates=1)

        # Test classification analysis
        class_analysis = optimizer._analyze_task(classification_dataset)
        assert "text" in class_analysis["input_keys"]
        assert "label" in class_analysis["output_keys"]
        assert "classification" in class_analysis["patterns"]
        assert "nlp_task" in class_analysis["patterns"]

        # Test QA analysis
        qa_analysis = optimizer._analyze_task(qa_dataset)
        assert "question" in qa_analysis["input_keys"]
        assert "context" in qa_analysis["input_keys"]
        assert "answer" in qa_analysis["output_keys"]
        assert "qa_task" in qa_analysis["patterns"]

        # Test transform analysis
        transform_analysis = optimizer._analyze_task(transform_dataset)
        assert "number" in transform_analysis["input_keys"]
        assert "doubled" in transform_analysis["output_keys"]
        assert "single_transform" in transform_analysis["patterns"]

    @pytest.mark.asyncio
    async def test_instruction_generation(self, classification_dataset):
        """Test instruction generation based on task analysis."""
        optimizer = InstructionOptimizer(metric=MockMetric(), num_candidates=5, seed=42)

        # Analyze task
        task_analysis = optimizer._analyze_task(classification_dataset)

        # Generate multiple instructions
        instructions = []
        for i in range(5):
            instruction = optimizer._generate_instruction(task_analysis, i)
            instructions.append(instruction)

        # All should be strings
        assert all(isinstance(inst, str) for inst in instructions)

        # Should mention relevant keys
        assert any("text" in inst for inst in instructions)
        assert any("label" in inst for inst in instructions)

        # Should have some variation
        assert len(set(instructions)) > 1

    @pytest.mark.asyncio
    async def test_selection_strategies(self, transform_dataset):
        """Test different selection strategies."""
        module = MockModule()
        metric = MockMetric()

        # Test best selection
        optimizer_best = InstructionOptimizer(
            metric=metric, num_candidates=5, selection_strategy="best", seed=42
        )
        result_best = await optimizer_best.optimize(module=module, dataset=transform_dataset)

        # Test weighted selection
        optimizer_weighted = InstructionOptimizer(
            metric=metric, num_candidates=5, selection_strategy="weighted", seed=42
        )
        result_weighted = await optimizer_weighted.optimize(
            module=module, dataset=transform_dataset
        )

        # Test tournament selection
        optimizer_tournament = InstructionOptimizer(
            metric=metric, num_candidates=5, selection_strategy="tournament", seed=42
        )
        result_tournament = await optimizer_tournament.optimize(
            module=module, dataset=transform_dataset
        )

        # All should produce valid results
        assert result_best.best_score >= 0
        assert result_weighted.best_score >= 0
        assert result_tournament.best_score >= 0

        # Check metadata
        assert result_best.metadata["selection_strategy"] == "best"
        assert result_weighted.metadata["selection_strategy"] == "weighted"
        assert result_tournament.metadata["selection_strategy"] == "tournament"

    @pytest.mark.asyncio
    async def test_custom_generator(self, classification_dataset):
        """Test with custom instruction generator."""

        def custom_generator(task_description="", examples=None, iteration=0):
            return f"Custom instruction {iteration}: Process the data carefully."

        module = MockModule()
        metric = MockMetric()
        optimizer = InstructionOptimizer(
            metric=metric, num_candidates=3, instruction_generator=custom_generator
        )

        result = await optimizer.optimize(module=module, dataset=classification_dataset)

        # Should use custom generator
        assert "Custom instruction" in result.metadata["best_instruction"]

    @pytest.mark.asyncio
    async def test_no_analysis(self, classification_dataset):
        """Test optimization without task analysis."""
        module = MockModule()
        metric = MockMetric()
        optimizer = InstructionOptimizer(
            metric=metric,
            num_candidates=3,
            analyze_examples=False,  # Disable analysis
            seed=42,
        )

        result = await optimizer.optimize(module=module, dataset=classification_dataset)

        # Should still work but with generic instructions
        assert "instruction" in result.optimized_module.parameters
        instruction = result.optimized_module.parameters["instruction"].value
        assert isinstance(instruction, str)

        # Task analysis should be empty
        assert result.metadata["task_analysis"] == {}

    @pytest.mark.asyncio
    async def test_empty_dataset(self):
        """Test handling of empty dataset."""
        module = MockModule()
        metric = MockMetric()
        optimizer = InstructionOptimizer(metric=metric, num_candidates=3)

        # Should handle empty dataset gracefully
        result = await optimizer.optimize(module=module, dataset=[])

        # Should generate generic instruction
        assert "instruction" in result.optimized_module.parameters
        instruction = result.optimized_module.parameters["instruction"].value
        assert "task" in instruction.lower()

    @pytest.mark.asyncio
    async def test_improvement_calculation(self, qa_dataset):
        """Test improvement calculation with validation set."""
        module = MockModule(behavior="quadratic")
        metric = MockMetric()
        optimizer = InstructionOptimizer(metric=metric, num_candidates=5, seed=42)

        # Split dataset
        train = qa_dataset[:2]
        val = qa_dataset[2:]

        result = await optimizer.optimize(module=module, dataset=train, validation_set=val)

        # Should calculate baseline and improvement
        assert "baseline_score" in result.metadata
        assert result.improvement == result.best_score - result.metadata["baseline_score"]

    @pytest.mark.asyncio
    async def test_pattern_specific_instructions(self):
        """Test that pattern-specific instructions are generated."""
        optimizer = InstructionOptimizer(metric=MockMetric(), num_candidates=10, seed=42)

        # Classification pattern
        class_analysis = {
            "input_keys": {"text"},
            "output_keys": {"label"},
            "patterns": ["classification", "nlp_task"],
        }

        class_instructions = []
        for i in range(5):
            inst = optimizer._generate_instruction(class_analysis, i)
            class_instructions.append(inst)

        # Should have classification-specific terms
        assert any(
            "classify" in inst.lower() or "categorize" in inst.lower()
            for inst in class_instructions
        )

        # QA pattern
        qa_analysis = {
            "input_keys": {"question", "context"},
            "output_keys": {"answer"},
            "patterns": ["qa_task"],
        }

        qa_instructions = []
        for i in range(5):
            inst = optimizer._generate_instruction(qa_analysis, i)
            qa_instructions.append(inst)

        # Should have QA-specific terms
        assert any(
            "answer" in inst.lower() or "question" in inst.lower() for inst in qa_instructions
        )

    @pytest.mark.asyncio
    async def test_reproducibility(self, classification_dataset):
        """Test reproducibility with same seed."""
        module = MockModule()
        metric = MockMetric()

        # Run twice with same seed
        optimizer1 = InstructionOptimizer(metric=metric, num_candidates=5, seed=999)
        result1 = await optimizer1.optimize(module=module, dataset=classification_dataset)

        optimizer2 = InstructionOptimizer(metric=metric, num_candidates=5, seed=999)
        result2 = await optimizer2.optimize(module=module, dataset=classification_dataset)

        # Should get same instruction (with same selection strategy)
        assert result1.metadata["best_instruction"] == result2.metadata["best_instruction"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
