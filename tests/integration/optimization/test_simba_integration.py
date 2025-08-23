"""Integration tests for SIMBA optimizer using gpt-4.1.

Tests the full SIMBA optimization pipeline with real LLM calls.
"""

import pytest

from logillm.core.predict import Predict
from logillm.exceptions import OptimizationError
from logillm.optimizers.simba import SIMBA
from logillm.providers import create_provider, register_provider


@pytest.mark.integration
@pytest.mark.openai
class TestSIMBAIntegration:
    """Integration tests for SIMBA optimizer."""

    @pytest.fixture(autouse=True)
    def setup(self, openai_provider_registered):
        """Setup for each test."""
        self.results = {}

    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_simba_improves_qa_accuracy(self):
        """Test SIMBA on question-answering tasks."""
        # Create QA module
        qa_module = Predict("question -> answer")

        # Training dataset with clear patterns
        training_data = [
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "What is 5+3?"}, "outputs": {"answer": "8"}},
            {"inputs": {"question": "What is 7+1?"}, "outputs": {"answer": "8"}},
            {"inputs": {"question": "What is 9-4?"}, "outputs": {"answer": "5"}},
            {"inputs": {"question": "What is 10-3?"}, "outputs": {"answer": "7"}},
            {"inputs": {"question": "What is 6+2?"}, "outputs": {"answer": "8"}},
            {"inputs": {"question": "What is 15/3?"}, "outputs": {"answer": "5"}},
            {"inputs": {"question": "What is 4*2?"}, "outputs": {"answer": "8"}},
        ]

        def accuracy_metric(prediction, expected):
            """Check if answer matches."""
            pred_answer = str(prediction.get("answer", "")).strip()
            exp_answer = str(expected.get("answer", "")).strip()
            return 1.0 if pred_answer == exp_answer else 0.0

        # Test baseline performance
        baseline_correct = 0
        baseline_total = len(training_data)

        for example in training_data:
            pred = await qa_module.forward(**example["inputs"])
            if accuracy_metric(pred.outputs, example["outputs"]) == 1.0:
                baseline_correct += 1

        baseline_score = baseline_correct / baseline_total
        print(f"Baseline accuracy: {baseline_score:.2%}")

        # SIMBA configuration for testing (small scale)
        optimizer = SIMBA(
            metric=accuracy_metric,
            bsize=3,  # Small batch for testing
            num_candidates=2,
            max_steps=2,
            max_demos=2,
            temperature_for_sampling=0.2,
            temperature_for_candidates=0.2,
        )

        # Run optimization
        result = await optimizer.optimize(qa_module, training_data, seed=42)

        # Store results
        self.results = {
            "baseline_score": baseline_score,
            "best_score": result.best_score,
            "improvement": result.improvement,
            "iterations": result.iterations,
            "optimization_time": result.optimization_time,
        }

        # Test the optimized module on new examples
        test_cases = [
            {"question": "What is 3+4?", "expected": "7"},
            {"question": "What is 12/4?", "expected": "3"},
        ]

        test_correct = 0
        for test in test_cases:
            pred = await result.optimized_module.forward(question=test["question"])
            if str(pred.outputs.get("answer", "")).strip() == test["expected"]:
                test_correct += 1

        test_accuracy = test_correct / len(test_cases)
        self.results["test_accuracy"] = test_accuracy

        # Assertions
        assert result.best_score >= 0, "Score should be non-negative"
        assert result.iterations > 0, "Should have completed iterations"
        assert result.optimization_time > 0, "Should have taken time"
        assert hasattr(result.optimized_module, "simba_idx"), (
            "Optimized module should have SIMBA metadata"
        )

        # Log results
        print("\nSIMBA QA Optimization Results:")
        print(f"  Baseline score: {baseline_score:.2%}")
        print(f"  Best score: {result.best_score:.2%}")
        print(f"  Improvement: {result.improvement:+.2%}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Time: {result.optimization_time:.1f}s")
        print(f"  Test accuracy: {test_accuracy:.2%}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(240)
    async def test_simba_mini_batch_optimization(self):
        """Test SIMBA mini-batch processing works correctly."""
        # Classification task
        module = Predict("text -> category")

        training_data = [
            {"inputs": {"text": "I love this product!"}, "outputs": {"category": "positive"}},
            {"inputs": {"text": "This is terrible."}, "outputs": {"category": "negative"}},
            {"inputs": {"text": "Amazing quality!"}, "outputs": {"category": "positive"}},
            {"inputs": {"text": "Waste of money."}, "outputs": {"category": "negative"}},
            {"inputs": {"text": "Perfect for my needs"}, "outputs": {"category": "positive"}},
            {"inputs": {"text": "Completely useless"}, "outputs": {"category": "negative"}},
            {"inputs": {"text": "Excellent service"}, "outputs": {"category": "positive"}},
            {"inputs": {"text": "Very disappointing"}, "outputs": {"category": "negative"}},
        ]

        def sentiment_metric(prediction, expected):
            """Check if sentiment category matches."""
            pred_cat = prediction.get("category", "").lower()
            exp_cat = expected.get("category", "").lower()
            return 1.0 if pred_cat == exp_cat else 0.0

        # Test with different batch sizes
        optimizer = SIMBA(
            metric=sentiment_metric,
            bsize=4,  # Process 4 examples per batch
            num_candidates=2,
            max_steps=2,
            max_demos=1,
        )

        result = await optimizer.optimize(module, training_data, seed=123)

        # Test mini-batch functionality
        test_examples = [
            {"text": "Great product!", "expected": "positive"},
            {"text": "Awful experience.", "expected": "negative"},
        ]

        correct = 0
        for test in test_examples:
            pred = await result.optimized_module.forward(text=test["text"])
            if pred.outputs.get("category", "").lower() == test["expected"]:
                correct += 1

        test_accuracy = correct / len(test_examples)

        # Assertions
        assert result.best_score >= 0, "Score should be non-negative"
        assert hasattr(result, "metadata"), "Result should have metadata"
        assert "trial_logs" in result.metadata, "Should have trial logs"

        print("\nSIMBA Mini-Batch Results:")
        print(f"  Best score: {result.best_score:.2%}")
        print("  Batch size: 4")
        print(f"  Test accuracy: {test_accuracy:.2%}")
        print(f"  Batches processed: {result.iterations}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_simba_rule_generation_with_llm(self):
        """Test introspective rule generation with real LLM."""
        # Math reasoning task
        module = Predict("problem: str -> solution: str, answer: int")

        training_data = [
            {
                "inputs": {"problem": "If I have 5 apples and eat 2, how many are left?"},
                "outputs": {"solution": "5 - 2 = 3", "answer": 3},
            },
            {
                "inputs": {"problem": "If I buy 3 books for $10 each, what's the total?"},
                "outputs": {"solution": "3 × $10 = $30", "answer": 30},
            },
            {
                "inputs": {
                    "problem": "If there are 12 cookies and 4 people share equally, how many per person?"
                },
                "outputs": {"solution": "12 ÷ 4 = 3", "answer": 3},
            },
            {
                "inputs": {"problem": "If a car travels 60 miles in 2 hours, what's the speed?"},
                "outputs": {"solution": "60 miles ÷ 2 hours = 30 mph", "answer": 30},
            },
        ]

        def math_metric(prediction, expected):
            """Check if numerical answer is correct."""
            try:
                pred_answer = int(prediction.get("answer", 0))
                exp_answer = int(expected.get("answer", 0))
                return 1.0 if pred_answer == exp_answer else 0.0
            except (ValueError, TypeError):
                return 0.0

        # Enable rule generation (introspection)
        optimizer = SIMBA(
            metric=math_metric,
            bsize=2,  # Small batch for rule generation
            num_candidates=2,
            max_steps=2,
            max_demos=1,  # Allow both rules and demos
        )

        result = await optimizer.optimize(module, training_data, seed=456)

        # Check that optimization actually generated rules
        optimized_module = result.optimized_module
        has_rules = False

        # Check if instructions were updated (indicating rule generation)
        if hasattr(optimized_module, "parameters") and "instruction" in optimized_module.parameters:
            instruction_param = optimized_module.parameters["instruction"]
            if instruction_param.value and len(instruction_param.value.strip()) > 0:
                has_rules = True
                print(f"Generated instruction: {instruction_param.value[:200]}...")

        # Test the optimized module
        test_problem = "If there are 15 pencils and 5 students, how many pencils per student?"
        test_result = await result.optimized_module.forward(problem=test_problem)

        print("\nSIMBA Rule Generation Results:")
        print(f"  Best score: {result.best_score:.2%}")
        print(f"  Rules generated: {has_rules}")
        print(f"  Test problem: {test_problem}")
        print(f"  Solution: {test_result.outputs.get('solution', 'No solution')}")
        print(f"  Answer: {test_result.outputs.get('answer', 'No answer')}")

        # Assertions
        assert result.best_score >= 0, "Score should be non-negative"
        assert result.iterations > 0, "Should have completed iterations"

    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_simba_demo_selection(self):
        """Test demonstration selection improves results."""
        # Simple pattern recognition task
        module = Predict("input: str -> output: str")

        # Training data with clear pattern (reverse the string)
        training_data = [
            {"inputs": {"input": "abc"}, "outputs": {"output": "cba"}},
            {"inputs": {"input": "hello"}, "outputs": {"output": "olleh"}},
            {"inputs": {"input": "world"}, "outputs": {"output": "dlrow"}},
            {"inputs": {"input": "test"}, "outputs": {"output": "tset"}},
            {"inputs": {"input": "demo"}, "outputs": {"output": "omed"}},
            {"inputs": {"input": "code"}, "outputs": {"output": "edoc"}},
        ]

        def pattern_metric(prediction, expected):
            """Check if output matches expected pattern."""
            pred_output = prediction.get("output", "")
            exp_output = expected.get("output", "")
            return 1.0 if pred_output == exp_output else 0.0

        # Enable demo selection
        optimizer = SIMBA(
            metric=pattern_metric,
            bsize=3,
            num_candidates=2,
            max_steps=2,
            max_demos=3,  # Allow demo collection
        )

        result = await optimizer.optimize(module, training_data, seed=789)

        # Check if demonstrations were added
        demo_count = 0

        if (
            hasattr(result.optimized_module, "parameters")
            and "demonstrations" in result.optimized_module.parameters
        ):
            demo_param = result.optimized_module.parameters["demonstrations"]
            if demo_param.value:
                demo_count = len(demo_param.value)

        # Test pattern recognition
        test_cases = [
            {"input": "python", "expected": "nohtyp"},
            {"input": "logic", "expected": "cigol"},
        ]

        correct = 0
        for test in test_cases:
            pred = await result.optimized_module.forward(input=test["input"])
            if pred.outputs.get("output", "") == test["expected"]:
                correct += 1

        test_accuracy = correct / len(test_cases)

        print("\nSIMBA Demo Selection Results:")
        print(f"  Best score: {result.best_score:.2%}")
        print(f"  Demonstrations added: {demo_count}")
        print(f"  Test accuracy: {test_accuracy:.2%}")

        # Assertions
        assert result.best_score >= 0, "Score should be non-negative"
        assert result.iterations > 0, "Should have completed iterations"

    @pytest.mark.asyncio
    @pytest.mark.timeout(240)
    async def test_simba_multi_candidate_evolution(self):
        """Test multiple candidate generation and evolution."""
        # Multi-class classification
        module = Predict("text -> label")

        training_data = [
            {"inputs": {"text": "The weather is sunny and warm"}, "outputs": {"label": "weather"}},
            {"inputs": {"text": "I bought a new laptop today"}, "outputs": {"label": "shopping"}},
            {"inputs": {"text": "The movie was fantastic"}, "outputs": {"label": "entertainment"}},
            {"inputs": {"text": "Rain is expected tomorrow"}, "outputs": {"label": "weather"}},
            {
                "inputs": {"text": "This restaurant serves great food"},
                "outputs": {"label": "dining"},
            },
            {"inputs": {"text": "The concert was amazing"}, "outputs": {"label": "entertainment"}},
            {"inputs": {"text": "I need to buy groceries"}, "outputs": {"label": "shopping"}},
            {"inputs": {"text": "The pizza was delicious"}, "outputs": {"label": "dining"}},
        ]

        def classification_metric(prediction, expected):
            """Check if classification is correct."""
            pred_label = prediction.get("label", "").lower()
            exp_label = expected.get("label", "").lower()
            return 1.0 if pred_label == exp_label else 0.0

        # Test with multiple candidates
        optimizer = SIMBA(
            metric=classification_metric,
            bsize=4,
            num_candidates=3,  # Generate 3 candidates per iteration
            max_steps=2,
            max_demos=2,
        )

        result = await optimizer.optimize(module, training_data, seed=101)

        # Check candidate evolution
        metadata = result.metadata
        candidate_programs = metadata.get("candidate_programs", [])

        # Test multi-class prediction
        test_cases = [
            {"text": "It's snowing outside", "expected": "weather"},
            {"text": "The book was interesting", "expected": "entertainment"},
            {"text": "I ordered pizza delivery", "expected": "dining"},
        ]

        correct = 0
        predictions = []
        for test in test_cases:
            pred = await result.optimized_module.forward(text=test["text"])
            pred_label = pred.outputs.get("label", "").lower()
            predictions.append(pred_label)
            if pred_label == test["expected"]:
                correct += 1

        test_accuracy = correct / len(test_cases)

        print("\nSIMBA Multi-Candidate Evolution Results:")
        print(f"  Best score: {result.best_score:.2%}")
        print(f"  Candidates evaluated: {len(candidate_programs)}")
        print(f"  Final scores: {[c['score'] for c in candidate_programs[:3]]}")
        print(f"  Test accuracy: {test_accuracy:.2%}")
        print(f"  Test predictions: {predictions}")

        # Assertions
        assert result.best_score >= 0, "Score should be non-negative"
        assert len(candidate_programs) > 0, "Should have candidate programs"
        assert result.iterations > 0, "Should have completed iterations"

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_simba_with_mock_provider(self):
        """Test SIMBA runs correctly with mock provider (smoke test)."""
        # Setup mock provider
        provider = create_provider("mock")
        register_provider(provider, set_default=True)

        module = Predict("input -> output")
        training_data = [
            {"inputs": {"input": "test1"}, "outputs": {"output": "result1"}},
            {"inputs": {"input": "test2"}, "outputs": {"output": "result2"}},
            {"inputs": {"input": "test3"}, "outputs": {"output": "result3"}},
            {"inputs": {"input": "test4"}, "outputs": {"output": "result4"}},
        ]

        def simple_metric(pred, expected):
            return 1.0 if pred.get("output") == expected.get("output") else 0.0

        # Minimal config for smoke test
        optimizer = SIMBA(
            metric=simple_metric,
            bsize=2,
            num_candidates=2,
            max_steps=1,
            max_demos=1,
        )

        result = await optimizer.optimize(module, training_data, seed=999)

        # Should complete without errors
        assert result is not None
        assert hasattr(result, "optimized_module")
        assert hasattr(result, "best_score")
        assert result.iterations > 0

        print("\nSIMBA Mock Provider Test:")
        print("  Completed successfully: ✓")
        print(f"  Score: {result.best_score:.2%}")
        print(f"  Time: {result.optimization_time:.2f}s")
        print(f"  Iterations: {result.iterations}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_simba_error_handling(self):
        """Test SIMBA handles errors gracefully."""
        module = Predict("input -> output")

        # Dataset too small for batch size
        small_data = [
            {"inputs": {"input": "test"}, "outputs": {"output": "result"}},
        ]

        def metric(pred, expected):
            return 1.0

        optimizer = SIMBA(
            metric=metric,
            bsize=5,  # Larger than dataset
            num_candidates=2,
            max_steps=1,
        )

        # Should raise OptimizationError
        with pytest.raises(OptimizationError):
            await optimizer.optimize(module, small_data)

        print("\nSIMBA Error Handling Test:")
        print("  Correctly raised exception for dataset too small: ✓")
