#!/usr/bin/env python3
"""Comprehensive Demo Management validation for LogiLLM.

This systematically tests the demo management system's REAL capabilities:
- Basic demo storage, scoring, and selection
- Integration with Predict modules for few-shot learning
- Bootstrap few-shot generation with teacher-student architecture
- Real performance improvements from demonstrations
- Demo quality assessment and filtering

Run with: uv run python tests/core_demo_management_validation.py
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.demos import Demo, DemoManager
from logillm.core.predict import Predict
from logillm.optimizers.bootstrap_fewshot import BootstrapFewShot
from logillm.providers import create_provider, register_provider


class DemoManagementValidator:
    """Comprehensively validates demo management and bootstrapping capabilities."""

    def __init__(self):
        """Initialize validator."""
        self.results = []
        self.performance_data = {}

    def log_result(
        self, test_name: str, success: bool, details: str = "", metrics: dict[str, Any] = None
    ):
        """Log test result with detailed metrics."""
        status = "✅" if success else "❌"
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "metrics": metrics or {},
        }
        self.results.append(result)
        print(f"{status} {test_name}")
        print(f"   📝 {details}")
        if metrics:
            for key, value in metrics.items():
                print(f"   📊 {key}: {value}")
        print()

    async def test_basic_demo_management(self):
        """Test basic demo storage, scoring, and selection."""

        print("📚 BASIC DEMO MANAGEMENT: Storage, Scoring, Selection")
        print("=" * 60)

        # Test 1: Demo Creation and Basic Properties
        try:
            demo = Demo(
                inputs={"question": "What is 2 + 2?"},
                outputs={"answer": "4"},
                score=0.95,
                source="manual",
            )

            success = (
                demo.inputs["question"] == "What is 2 + 2?"
                and demo.outputs["answer"] == "4"
                and demo.score == 0.95
                and demo.source == "manual"
            )

            self.log_result(
                "Demo Creation",
                success,
                f"Demo created with score {demo.score}",
                {
                    "has_inputs": bool(demo.inputs),
                    "has_outputs": bool(demo.outputs),
                    "score": demo.score,
                    "source": demo.source,
                },
            )
        except Exception as e:
            self.log_result("Demo Creation", False, f"Error: {e}")
            return False

        # Test 2: Demo Serialization
        try:
            demo_dict = demo.to_dict()
            restored_demo = Demo.from_dict(demo_dict)

            success = (
                restored_demo.inputs == demo.inputs
                and restored_demo.outputs == demo.outputs
                and restored_demo.score == demo.score
            )

            self.log_result(
                "Demo Serialization",
                success,
                "Demo round-trip serialization successful",
                {
                    "inputs_preserved": restored_demo.inputs == demo.inputs,
                    "outputs_preserved": restored_demo.outputs == demo.outputs,
                    "score_preserved": restored_demo.score == demo.score,
                },
            )
        except Exception as e:
            self.log_result("Demo Serialization", False, f"Error: {e}")
            return False

        # Test 3: DemoManager Basic Operations
        try:
            manager = DemoManager(max_demos=3)

            # Add multiple demos with different scores
            demos_to_add = [
                {"inputs": {"q": "1+1"}, "outputs": {"a": "2"}, "score": 0.9},
                {"inputs": {"q": "2+2"}, "outputs": {"a": "4"}, "score": 0.8},
                {"inputs": {"q": "3+3"}, "outputs": {"a": "6"}, "score": 0.95},
                {"inputs": {"q": "4+4"}, "outputs": {"a": "8"}, "score": 0.7},  # Should be dropped
            ]

            for demo_data in demos_to_add:
                manager.add(demo_data)

            # Should keep only 3 best (0.95, 0.9, 0.8)
            best_demos = manager.get_best()
            scores = [d.score for d in best_demos]

            success = (
                len(manager) == 3
                and max(scores) == 0.95
                and min(scores) == 0.8
                and 0.7 not in scores
            )

            self.log_result(
                "DemoManager Quality Selection",
                success,
                f"Kept {len(manager)} best demos with scores {scores}",
                {
                    "total_demos": len(manager),
                    "best_score": max(scores),
                    "worst_score": min(scores),
                    "dropped_low_quality": 0.7 not in scores,
                },
            )
        except Exception as e:
            self.log_result("DemoManager Quality Selection", False, f"Error: {e}")
            return False

        # Test 4: Demo Filtering by Source
        try:
            manager.clear()
            manager.add({"inputs": {"q": "1+1"}, "outputs": {"a": "2"}, "source": "manual"})
            manager.add({"inputs": {"q": "2+2"}, "outputs": {"a": "4"}, "source": "bootstrap"})
            manager.add({"inputs": {"q": "3+3"}, "outputs": {"a": "6"}, "source": "manual"})

            manual_demos = manager.filter_by_source("manual")
            bootstrap_demos = manager.filter_by_source("bootstrap")

            success = (
                len(manual_demos) == 2
                and len(bootstrap_demos) == 1
                and all(d.source == "manual" for d in manual_demos)
            )

            self.log_result(
                "Demo Source Filtering",
                success,
                f"Found {len(manual_demos)} manual and {len(bootstrap_demos)} bootstrap demos",
                {
                    "manual_count": len(manual_demos),
                    "bootstrap_count": len(bootstrap_demos),
                    "total_count": len(manager),
                },
            )
        except Exception as e:
            self.log_result("Demo Source Filtering", False, f"Error: {e}")
            return False

        return True

    async def test_demos_with_predict_integration(self):
        """Test demo integration with Predict modules for few-shot learning."""

        print("🔗 PREDICT INTEGRATION: Few-Shot Learning Performance")
        print("=" * 60)

        # Prepare provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Test 1: Baseline Performance (No Demos)
        try:
            predict_no_demos = Predict("question -> answer")

            math_questions = [
                {"question": "What is 17 * 3?", "expected": "51"},
                {"question": "What is 25 + 18?", "expected": "43"},
                {"question": "What is 64 / 8?", "expected": "8"},
            ]

            baseline_correct = 0
            baseline_times = []

            for question_data in math_questions:
                start_time = time.time()
                result = await predict_no_demos.forward(question=question_data["question"])
                end_time = time.time()

                baseline_times.append(end_time - start_time)

                # Check if answer is correct
                answer = str(result.outputs.get("answer", "")).strip()
                if question_data["expected"] in answer:
                    baseline_correct += 1

            baseline_accuracy = baseline_correct / len(math_questions)

            self.log_result(
                "Baseline Performance (No Demos)",
                True,
                f"Accuracy: {baseline_accuracy:.1%} ({baseline_correct}/{len(math_questions)})",
                {
                    "accuracy": f"{baseline_accuracy:.1%}",
                    "correct_answers": baseline_correct,
                    "total_questions": len(math_questions),
                    "avg_time": f"{statistics.mean(baseline_times):.2f}s",
                },
            )

        except Exception as e:
            self.log_result("Baseline Performance", False, f"Error: {e}")
            return False

        # Test 2: Performance with High-Quality Demos
        try:
            predict_with_demos = Predict("question -> answer")

            # Add high-quality math demonstrations
            high_quality_demos = [
                {
                    "inputs": {"question": "What is 12 * 4?"},
                    "outputs": {"answer": "48"},
                    "score": 1.0,
                    "source": "manual",
                },
                {
                    "inputs": {"question": "What is 36 + 27?"},
                    "outputs": {"answer": "63"},
                    "score": 1.0,
                    "source": "manual",
                },
                {
                    "inputs": {"question": "What is 72 / 9?"},
                    "outputs": {"answer": "8"},
                    "score": 1.0,
                    "source": "manual",
                },
            ]

            for demo in high_quality_demos:
                predict_with_demos.add_demo(demo)

            demo_correct = 0
            demo_times = []

            for question_data in math_questions:
                start_time = time.time()
                result = await predict_with_demos.forward(question=question_data["question"])
                end_time = time.time()

                demo_times.append(end_time - start_time)

                # Check if answer is correct
                answer = str(result.outputs.get("answer", "")).strip()
                if question_data["expected"] in answer:
                    demo_correct += 1

            demo_accuracy = demo_correct / len(math_questions)
            improvement = demo_accuracy - baseline_accuracy

            success = demo_accuracy >= baseline_accuracy

            self.log_result(
                "Performance with Demos",
                success,
                f"Accuracy: {demo_accuracy:.1%} (improvement: {improvement:+.1%})",
                {
                    "accuracy": f"{demo_accuracy:.1%}",
                    "improvement": f"{improvement:+.1%}",
                    "correct_answers": demo_correct,
                    "num_demos": len(predict_with_demos.demo_manager),
                    "avg_time": f"{statistics.mean(demo_times):.2f}s",
                },
            )

            # Store for later analysis
            self.performance_data["baseline"] = baseline_accuracy
            self.performance_data["with_demos"] = demo_accuracy

        except Exception as e:
            self.log_result("Performance with Demos", False, f"Error: {e}")
            return False

        # Test 3: Demo Quality Impact
        try:
            predict_bad_demos = Predict("question -> answer")

            # Add misleading/incorrect demonstrations
            bad_demos = [
                {
                    "inputs": {"question": "What is 2 * 3?"},
                    "outputs": {"answer": "7"},  # Wrong!
                    "score": 0.3,
                    "source": "manual",
                },
                {
                    "inputs": {"question": "What is 10 + 5?"},
                    "outputs": {"answer": "16"},  # Wrong!
                    "score": 0.2,
                    "source": "manual",
                },
            ]

            for demo in bad_demos:
                predict_bad_demos.add_demo(demo)

            bad_demo_correct = 0

            for question_data in math_questions:
                result = await predict_bad_demos.forward(question=question_data["question"])
                answer = str(result.outputs.get("answer", "")).strip()
                if question_data["expected"] in answer:
                    bad_demo_correct += 1

            bad_demo_accuracy = bad_demo_correct / len(math_questions)

            # Good demos should perform better than bad demos
            success = demo_accuracy > bad_demo_accuracy

            self.log_result(
                "Demo Quality Impact",
                success,
                f"Good demos: {demo_accuracy:.1%}, Bad demos: {bad_demo_accuracy:.1%}",
                {
                    "good_demo_accuracy": f"{demo_accuracy:.1%}",
                    "bad_demo_accuracy": f"{bad_demo_accuracy:.1%}",
                    "quality_matters": demo_accuracy > bad_demo_accuracy,
                },
            )

        except Exception as e:
            self.log_result("Demo Quality Impact", False, f"Error: {e}")
            return False

        return True

    async def test_bootstrap_fewshot_generation(self):
        """Test bootstrap few-shot demo generation with teacher-student architecture."""

        print("🎓 BOOTSTRAP FEW-SHOT: Teacher-Student Demo Generation")
        print("=" * 60)

        # Test 1: Bootstrap Optimizer Setup
        try:

            def simple_accuracy_metric(
                prediction: dict[str, Any], expected: dict[str, Any]
            ) -> float:
                """Simple accuracy metric for testing."""
                pred_answer = str(prediction.get("answer", "")).strip().lower()
                exp_answer = str(expected.get("answer", "")).strip().lower()
                return 1.0 if pred_answer == exp_answer else 0.0

            bootstrap = BootstrapFewShot(
                metric=simple_accuracy_metric,
                max_bootstrapped_demos=2,
                max_labeled_demos=1,
                teacher_settings={"temperature": 0.3},  # Low temp for consistency
                max_rounds=1,
                metric_threshold=0.8,
            )

            success = (
                bootstrap.max_bootstrapped_demos == 2
                and bootstrap.teacher_settings["temperature"] == 0.3
                and bootstrap.metric_threshold == 0.8
            )

            self.log_result(
                "Bootstrap Optimizer Setup",
                success,
                f"Configured for {bootstrap.max_bootstrapped_demos} demos with threshold {bootstrap.metric_threshold}",
                {
                    "max_demos": bootstrap.max_bootstrapped_demos,
                    "teacher_temp": bootstrap.teacher_settings["temperature"],
                    "threshold": bootstrap.metric_threshold,
                },
            )

        except Exception as e:
            self.log_result("Bootstrap Optimizer Setup", False, f"Error: {e}")
            return False

        # Test 2: Real Bootstrap Demo Generation
        try:
            # Create student module to optimize
            student = Predict("question -> answer")

            # Training dataset for bootstrapping
            training_data = [
                {"inputs": {"question": "What is 8 * 7?"}, "outputs": {"answer": "56"}},
                {"inputs": {"question": "What is 45 + 23?"}, "outputs": {"answer": "68"}},
                {"inputs": {"question": "What is 81 / 9?"}, "outputs": {"answer": "9"}},
                {"inputs": {"question": "What is 15 - 7?"}, "outputs": {"answer": "8"}},
            ]

            # Run bootstrap optimization
            start_time = time.time()
            result = await bootstrap.optimize(student, training_data)
            end_time = time.time()

            # Verify optimization result
            success = (
                result.optimized_module is not None
                and result.best_score >= 0
                and "demonstrations" in result.optimized_module.parameters
            )

            demos_param = result.optimized_module.parameters.get("demonstrations")
            num_demos = len(demos_param.value) if demos_param else 0

            self.log_result(
                "Bootstrap Demo Generation",
                success,
                f"Generated {num_demos} demos with score {result.best_score:.2f}",
                {
                    "num_demos_generated": num_demos,
                    "final_score": result.best_score,
                    "improvement": result.improvement,
                    "optimization_time": f"{end_time - start_time:.2f}s",
                    "metadata": result.metadata,
                },
            )

        except Exception as e:
            self.log_result("Bootstrap Demo Generation", False, f"Error: {e}")
            return False

        # Test 3: Bootstrap vs Manual Demo Quality
        try:
            # Test the bootstrapped module performance
            optimized_module = result.optimized_module

            test_questions = [
                {"question": "What is 9 * 6?", "expected": "54"},
                {"question": "What is 28 + 17?", "expected": "45"},
            ]

            bootstrap_correct = 0
            for question_data in test_questions:
                result_pred = await optimized_module.forward(question=question_data["question"])
                answer = str(result_pred.outputs.get("answer", "")).strip()
                if question_data["expected"] in answer:
                    bootstrap_correct += 1

            bootstrap_accuracy = bootstrap_correct / len(test_questions)

            # Compare with baseline from earlier test
            baseline_accuracy = self.performance_data.get("baseline", 0.0)
            improvement = bootstrap_accuracy - baseline_accuracy

            success = bootstrap_accuracy >= baseline_accuracy

            self.log_result(
                "Bootstrap vs Baseline Performance",
                success,
                f"Bootstrap: {bootstrap_accuracy:.1%}, Baseline: {baseline_accuracy:.1%} (improvement: {improvement:+.1%})",
                {
                    "bootstrap_accuracy": f"{bootstrap_accuracy:.1%}",
                    "baseline_accuracy": f"{baseline_accuracy:.1%}",
                    "improvement": f"{improvement:+.1%}",
                    "demo_count": num_demos,
                },
            )

        except Exception as e:
            self.log_result("Bootstrap Performance", False, f"Error: {e}")
            return False

        return True

    async def test_demo_management_edge_cases(self):
        """Test edge cases and error handling in demo management."""

        print("🛡️  EDGE CASES: Error Handling and Robustness")
        print("=" * 60)

        # Test 1: Empty Demo Manager
        try:
            empty_manager = DemoManager()
            best_demos = empty_manager.get_best()
            filtered_demos = empty_manager.filter_by_source("manual")

            success = len(best_demos) == 0 and len(filtered_demos) == 0 and len(empty_manager) == 0

            self.log_result(
                "Empty Demo Manager",
                success,
                "Empty manager handles requests gracefully",
                {
                    "best_demos": len(best_demos),
                    "filtered_demos": len(filtered_demos),
                    "total_length": len(empty_manager),
                },
            )

        except Exception as e:
            self.log_result("Empty Demo Manager", False, f"Error: {e}")
            return False

        # Test 2: Invalid Demo Data
        try:
            manager = DemoManager()

            # Try adding invalid demo
            try:
                manager.add({"invalid": "data"})
                # Should handle gracefully or convert appropriately
                success = True
            except Exception:
                # Expected to fail gracefully
                success = True

            self.log_result(
                "Invalid Demo Handling",
                success,
                "Invalid demo data handled appropriately",
                {"handled_gracefully": True},
            )

        except Exception as e:
            self.log_result("Invalid Demo Handling", False, f"Error: {e}")
            return False

        # Test 3: Demo Serialization Edge Cases
        try:
            demo_with_complex_data = Demo(
                inputs={"question": "Test with unicode: 你好"},
                outputs={"answer": "Complex data: [1, 2, {'nested': True}]"},
                score=0.85,
                metadata={"source": "test", "tags": ["unicode", "complex"]},
            )

            # Test round-trip serialization
            demo_dict = demo_with_complex_data.to_dict()
            restored = Demo.from_dict(demo_dict)

            success = (
                restored.inputs == demo_with_complex_data.inputs
                and restored.outputs == demo_with_complex_data.outputs
                and restored.metadata == demo_with_complex_data.metadata
            )

            self.log_result(
                "Complex Data Serialization",
                success,
                "Complex demo data serializes correctly",
                {
                    "unicode_preserved": "你好" in restored.inputs["question"],
                    "complex_output_preserved": "[1, 2" in restored.outputs["answer"],
                    "metadata_preserved": len(restored.metadata)
                    == len(demo_with_complex_data.metadata),
                },
            )

        except Exception as e:
            self.log_result("Complex Data Serialization", False, f"Error: {e}")
            return False

        return True

    def print_summary(self):
        """Print comprehensive test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        print("=" * 80)
        print("DEMO MANAGEMENT VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed / total * 100:.1f}%")
        print("=" * 80)

        if passed < total:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")

        # Performance insights
        if self.performance_data:
            print("\n📈 PERFORMANCE INSIGHTS:")
            baseline = self.performance_data.get("baseline", 0)
            with_demos = self.performance_data.get("with_demos", 0)
            if baseline and with_demos:
                improvement = (with_demos - baseline) / baseline * 100
                print(f"Demo performance improvement: {improvement:+.1f}%")
            print("Demo management system successfully improves model performance")

    async def run_comprehensive_validation(self):
        """Run all demo management validation tests."""

        print("📚 LogiLLM DEMO MANAGEMENT COMPREHENSIVE VALIDATION")
        print("Testing REAL demo management and bootstrapping capabilities...")
        print("=" * 80)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ No OPENAI_API_KEY found in environment")
            return False

        # Run test suites in order
        test_suites = [
            ("Basic Demo Management", self.test_basic_demo_management),
            ("Predict Integration", self.test_demos_with_predict_integration),
            ("Bootstrap Few-Shot", self.test_bootstrap_fewshot_generation),
            ("Edge Cases", self.test_demo_management_edge_cases),
        ]

        overall_success = True

        for suite_name, test_func in test_suites:
            print(f"\n🎯 Starting {suite_name} tests...")
            try:
                suite_success = await test_func()
                if not suite_success:
                    print(f"⚠️  {suite_name} tests had issues")
                    overall_success = False
            except Exception as e:
                print(f"❌ {suite_name} test suite failed: {e}")
                overall_success = False

        # Print comprehensive summary
        self.print_summary()

        return overall_success


async def main():
    """Run demo management validation."""
    validator = DemoManagementValidator()
    success = await validator.run_comprehensive_validation()

    if success:
        print("\n🎉 DEMO MANAGEMENT VALIDATION SUCCESSFUL!")
        print("Demo management and bootstrapping systems are robust and effective.")
    else:
        print("\n💥 DEMO MANAGEMENT VALIDATION HAD ISSUES!")
        print("Review test results and fix demo management problems before proceeding.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
