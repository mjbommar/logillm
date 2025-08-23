#!/usr/bin/env python3
"""Comprehensive Bootstrap Few-Shot validation for LogiLLM.

This systematically tests bootstrap few-shot optimization with REAL example generation:
- Teacher-student architecture with actual LLM-generated examples
- Quality assessment and filtering of generated demonstrations
- Bootstrap optimization performance improvements over baseline
- Real example generation across different task types
- Demo selection and quality scoring with actual metrics
- Bootstrapping convergence and effectiveness analysis

Run with: uv run python tests/optimization_bootstrap_fewshot_validation.py
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.predict import Predict
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider


class BootstrapFewShotValidator:
    """Comprehensively validates bootstrap few-shot optimization with real example generation."""

    def __init__(self):
        """Initialize validator."""
        self.results = []
        self.optimization_data = {}

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

    async def test_basic_bootstrap_functionality(self):
        """Test basic bootstrap few-shot setup and configuration."""

        print("🎓 BASIC BOOTSTRAP FEW-SHOT: Configuration and Setup")
        print("=" * 60)

        # Prepare provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Test 1: Bootstrap Optimizer Configuration
        try:

            def simple_accuracy_metric(
                prediction: dict[str, Any], expected: dict[str, Any]
            ) -> float:
                """Simple accuracy metric for bootstrap testing."""
                pred_answer = str(prediction.get("answer", "")).lower().strip()
                exp_answer = str(expected.get("answer", "")).lower().strip()
                # Flexible matching for bootstrap-generated examples
                return 1.0 if exp_answer in pred_answer or pred_answer in exp_answer else 0.0

            bootstrap_optimizer = BootstrapFewShot(
                metric=simple_accuracy_metric,
                max_bootstrapped_demos=3,  # Reduced for speed
                max_labeled_demos=2,
                teacher_settings={
                    "temperature": 0.2,
                    "max_tokens": 150,
                },  # Conservative for quality
                max_rounds=2,  # Reduced for speed
                metric_threshold=0.6,
            )

            success = (
                bootstrap_optimizer.max_bootstrapped_demos == 3
                and bootstrap_optimizer.teacher_settings["temperature"] == 0.2
                and bootstrap_optimizer.metric_threshold == 0.6
            )

            self.log_result(
                "Bootstrap Optimizer Configuration",
                success,
                f"Configured for {bootstrap_optimizer.max_bootstrapped_demos} demos",
                {
                    "max_bootstrapped_demos": bootstrap_optimizer.max_bootstrapped_demos,
                    "max_labeled_demos": bootstrap_optimizer.max_labeled_demos,
                    "teacher_temperature": bootstrap_optimizer.teacher_settings["temperature"],
                    "max_rounds": bootstrap_optimizer.max_rounds,
                    "metric_threshold": bootstrap_optimizer.metric_threshold,
                },
            )

        except Exception as e:
            self.log_result("Bootstrap Optimizer Configuration", False, f"Error: {e}")
            return False

        # Test 2: Teacher-Student Module Creation
        try:
            # Create student module
            student_module = Predict("question -> answer")

            # Test teacher module creation manually (as done in optimize method)
            import copy

            teacher_module = copy.deepcopy(student_module)
            # Apply teacher settings
            if hasattr(teacher_module, "config"):
                teacher_module.config = teacher_module.config or {}
                teacher_module.config.update(bootstrap_optimizer.teacher_settings)

            success = (
                teacher_module is not None
                and hasattr(teacher_module, "signature")
                and str(teacher_module.signature) == str(student_module.signature)
            )

            self.log_result(
                "Teacher-Student Module Creation",
                success,
                f"Teacher module created with signature: {teacher_module.signature}",
                {
                    "teacher_created": teacher_module is not None,
                    "signature_match": str(teacher_module.signature)
                    == str(student_module.signature),
                    "student_signature": str(student_module.signature),
                    "teacher_config": teacher_module.config
                    if hasattr(teacher_module, "config")
                    else None,
                },
            )

        except Exception as e:
            self.log_result("Teacher-Student Module Creation", False, f"Error: {e}")
            return False

        return True

    async def test_bootstrap_example_generation(self):
        """Test real bootstrap example generation with teacher model."""

        print("🏭 BOOTSTRAP EXAMPLE GENERATION: Teacher Model Creates Examples")
        print("=" * 60)

        def math_accuracy_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            """Math-focused accuracy metric."""
            pred_answer = str(prediction.get("answer", "")).strip()
            exp_answer = str(expected.get("answer", "")).strip()

            # Extract numbers for comparison
            import re

            pred_nums = re.findall(r"\d+", pred_answer)
            exp_nums = re.findall(r"\d+", exp_answer)

            if pred_nums and exp_nums:
                return 1.0 if pred_nums[0] == exp_nums[0] else 0.0

            return 1.0 if exp_answer in pred_answer else 0.0

        # Test 1: Single Bootstrap Round
        try:
            bootstrap_gen = BootstrapFewShot(
                metric=math_accuracy_metric,
                max_bootstrapped_demos=2,  # Small for testing
                max_labeled_demos=1,
                teacher_settings={"temperature": 0.3},  # Balanced for diversity
                max_rounds=1,  # Single round test
                metric_threshold=0.5,
            )

            student_module = Predict("problem -> answer")

            # Minimal training dataset to bootstrap from
            training_data = [
                {"inputs": {"problem": "What is 9 + 7?"}, "outputs": {"answer": "16"}},
                {"inputs": {"problem": "What is 15 * 4?"}, "outputs": {"answer": "60"}},
                {"inputs": {"problem": "What is 72 / 8?"}, "outputs": {"answer": "9"}},
            ]

            start_time = time.time()
            result = await bootstrap_gen.optimize(student_module, training_data)
            end_time = time.time()

            # Check if examples were generated - they're stored as parameters in BootstrapFewShot
            optimized_demos = []
            bootstrap_demos = []
            if (
                hasattr(result.optimized_module, "parameters")
                and "demonstrations" in result.optimized_module.parameters
            ):
                demo_param = result.optimized_module.parameters["demonstrations"]
                demo_dicts = demo_param.value if hasattr(demo_param, "value") else []
                optimized_demos = demo_dicts
                # Check metadata to find bootstrap demos
                for demo_dict in demo_dicts:
                    if isinstance(demo_dict, dict) and demo_dict.get("metadata", {}).get(
                        "teacher", False
                    ):
                        bootstrap_demos.append(demo_dict)

            success = (
                result.best_score >= 0 and len(bootstrap_demos) > 0 and result.improvement >= 0
            )

            self.log_result(
                "Single Bootstrap Round",
                success,
                f"Generated {len(bootstrap_demos)} bootstrap examples with score {result.best_score:.2f}",
                {
                    "bootstrap_examples_generated": len(bootstrap_demos),
                    "total_demos": len(optimized_demos),
                    "final_score": result.best_score,
                    "improvement": result.improvement,
                    "optimization_time": f"{end_time - start_time:.2f}s",
                    "rounds_completed": result.iterations,
                },
            )

            # Store for analysis
            self.optimization_data["single_bootstrap"] = {
                "result": result,
                "bootstrap_demos": bootstrap_demos,
                "total_demos": optimized_demos,
            }

        except Exception as e:
            self.log_result("Single Bootstrap Round", False, f"Error: {e}")
            return False

        # Test 2: Bootstrap Example Quality Assessment
        try:
            if bootstrap_demos:
                # Analyze quality of generated examples
                quality_scores = [demo.get("score", 0.0) for demo in bootstrap_demos]
                avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0

                # Check example diversity
                unique_inputs = set()
                for demo in bootstrap_demos:
                    inputs_dict = demo.get("inputs", {})
                    input_str = str(inputs_dict.get("problem", ""))
                    unique_inputs.add(input_str)

                diversity_ratio = (
                    len(unique_inputs) / len(bootstrap_demos) if bootstrap_demos else 0.0
                )

                success = avg_quality > 0.0 and diversity_ratio > 0.5

                self.log_result(
                    "Bootstrap Example Quality Assessment",
                    success,
                    f"Average quality: {avg_quality:.2f}, Diversity: {diversity_ratio:.2f}",
                    {
                        "average_quality_score": avg_quality,
                        "quality_scores": quality_scores,
                        "diversity_ratio": diversity_ratio,
                        "unique_inputs": len(unique_inputs),
                        "total_examples": len(bootstrap_demos),
                    },
                )
            else:
                self.log_result(
                    "Bootstrap Example Quality Assessment", False, "No bootstrap examples to assess"
                )

        except Exception as e:
            self.log_result("Bootstrap Example Quality Assessment", False, f"Error: {e}")
            return False

        return True

    async def test_bootstrap_performance_improvement(self):
        """Test bootstrap optimization performance improvements."""

        print("📈 BOOTSTRAP PERFORMANCE: Improvement Over Baseline")
        print("=" * 60)

        def precise_math_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            """Precise metric for performance comparison."""
            pred_answer = str(prediction.get("answer", "")).strip()
            exp_answer = str(expected.get("answer", "")).strip()

            # Try exact match first
            if pred_answer == exp_answer:
                return 1.0

            # Extract and compare numbers
            import re

            pred_nums = re.findall(r"\d+", pred_answer)
            exp_nums = re.findall(r"\d+", exp_answer)

            if pred_nums and exp_nums and pred_nums[0] == exp_nums[0]:
                return 1.0

            return 0.0

        # Test dataset for performance comparison
        test_dataset = [
            {"inputs": {"problem": "Calculate 23 + 34"}, "outputs": {"answer": "57"}},
            {"inputs": {"problem": "Calculate 8 * 12"}, "outputs": {"answer": "96"}},
            {"inputs": {"problem": "Calculate 84 - 29"}, "outputs": {"answer": "55"}},
            {"inputs": {"problem": "Calculate 144 / 12"}, "outputs": {"answer": "12"}},
        ]

        # Test 1: Baseline Performance (No Bootstrap)
        try:
            baseline_module = Predict("problem -> answer")

            baseline_correct = 0
            baseline_times = []

            for test_case in test_dataset:
                start_time = time.time()
                result = await baseline_module.forward(**test_case["inputs"])
                end_time = time.time()

                baseline_times.append(end_time - start_time)
                score = precise_math_metric(result.outputs, test_case["outputs"])
                baseline_correct += score

            baseline_accuracy = baseline_correct / len(test_dataset)

            self.log_result(
                "Baseline Performance (No Bootstrap)",
                True,
                f"Accuracy: {baseline_accuracy:.1%} ({baseline_correct}/{len(test_dataset)})",
                {
                    "accuracy": f"{baseline_accuracy:.1%}",
                    "correct_answers": baseline_correct,
                    "total_questions": len(test_dataset),
                    "avg_time": f"{statistics.mean(baseline_times):.2f}s",
                },
            )

            # Store baseline for comparison
            self.optimization_data["baseline_performance"] = baseline_accuracy

        except Exception as e:
            self.log_result("Baseline Performance", False, f"Error: {e}")
            return False

        # Test 2: Bootstrap Optimization Performance
        try:
            bootstrap_perf = BootstrapFewShot(
                metric=precise_math_metric,
                max_bootstrapped_demos=3,
                max_labeled_demos=2,
                teacher_settings={"temperature": 0.2},  # Conservative for accuracy
                student_settings={"temperature": 0.5},
                max_rounds=2,
                metric_threshold=0.7,
            )

            # Training data for bootstrapping
            training_data = [
                {"inputs": {"problem": "Calculate 15 + 18"}, "outputs": {"answer": "33"}},
                {"inputs": {"problem": "Calculate 7 * 9"}, "outputs": {"answer": "63"}},
                {"inputs": {"problem": "Calculate 90 - 35"}, "outputs": {"answer": "55"}},
                {"inputs": {"problem": "Calculate 48 / 6"}, "outputs": {"answer": "8"}},
            ]

            student_for_bootstrap = Predict("problem -> answer")

            start_time = time.time()
            bootstrap_result = await bootstrap_perf.optimize(student_for_bootstrap, training_data)
            end_time = time.time()

            # Test optimized module on test dataset
            optimized_module = bootstrap_result.optimized_module
            bootstrap_correct = 0
            bootstrap_times = []

            for test_case in test_dataset:
                start_time_test = time.time()
                result = await optimized_module.forward(**test_case["inputs"])
                end_time_test = time.time()

                bootstrap_times.append(end_time_test - start_time_test)
                score = precise_math_metric(result.outputs, test_case["outputs"])
                bootstrap_correct += score

            bootstrap_accuracy = bootstrap_correct / len(test_dataset)
            improvement = bootstrap_accuracy - baseline_accuracy

            success = bootstrap_accuracy >= baseline_accuracy

            self.log_result(
                "Bootstrap Optimization Performance",
                success,
                f"Accuracy: {bootstrap_accuracy:.1%} (improvement: {improvement:+.1%})",
                {
                    "bootstrap_accuracy": f"{bootstrap_accuracy:.1%}",
                    "baseline_accuracy": f"{baseline_accuracy:.1%}",
                    "improvement": f"{improvement:+.1%}",
                    "optimization_time": f"{end_time - start_time:.2f}s",
                    "bootstrap_score": bootstrap_result.best_score,
                    "generated_demos": len(
                        optimized_module.parameters.get("demonstrations", {}).value
                    )
                    if hasattr(optimized_module, "parameters")
                    and "demonstrations" in optimized_module.parameters
                    else 0,
                },
            )

            # Store for analysis
            self.optimization_data["bootstrap_performance"] = {
                "accuracy": bootstrap_accuracy,
                "improvement": improvement,
                "result": bootstrap_result,
            }

        except Exception as e:
            self.log_result("Bootstrap Optimization Performance", False, f"Error: {e}")
            return False

        return True

    async def test_multi_task_bootstrapping(self):
        """Test bootstrap few-shot across different task types."""

        print("🎯 MULTI-TASK BOOTSTRAPPING: Performance Across Task Types")
        print("=" * 60)

        task_types = [
            {
                "name": "Arithmetic",
                "signature": "expression -> result",
                "metric": lambda pred, exp: 1.0
                if str(exp.get("result", "")).strip() in str(pred.get("result", "")).strip()
                else 0.0,
                "training_data": [
                    {"inputs": {"expression": "25 + 17"}, "outputs": {"result": "42"}},
                    {"inputs": {"expression": "8 * 11"}, "outputs": {"result": "88"}},
                ],
                "test_data": [
                    {"inputs": {"expression": "19 + 24"}, "outputs": {"result": "43"}},
                    {"inputs": {"expression": "6 * 13"}, "outputs": {"result": "78"}},
                ],
            },
            {
                "name": "Simple Q&A",
                "signature": "question -> answer",
                "metric": lambda pred, exp: 1.0
                if str(exp.get("answer", "")).lower() in str(pred.get("answer", "")).lower()
                else 0.0,
                "training_data": [
                    {
                        "inputs": {"question": "What is the capital of France?"},
                        "outputs": {"answer": "Paris"},
                    },
                    {
                        "inputs": {"question": "What color is the sky?"},
                        "outputs": {"answer": "blue"},
                    },
                ],
                "test_data": [
                    {
                        "inputs": {"question": "What is the capital of Italy?"},
                        "outputs": {"answer": "Rome"},
                    },
                    {
                        "inputs": {"question": "What color is grass?"},
                        "outputs": {"answer": "green"},
                    },
                ],
            },
        ]

        task_results = {}

        for task in task_types:
            try:
                print(f"\n🧪 Testing: {task['name']}")

                # Create bootstrap optimizer for this task
                task_bootstrap = BootstrapFewShot(
                    metric=task["metric"],
                    max_bootstrapped_demos=2,  # Small for speed
                    max_labeled_demos=1,
                    teacher_settings={"temperature": 0.3},
                    max_rounds=1,
                    metric_threshold=0.5,
                )

                # Test baseline performance
                baseline_module = Predict(task["signature"])
                baseline_scores = []

                for test_case in task["test_data"]:
                    result = await baseline_module.forward(**test_case["inputs"])
                    score = task["metric"](result.outputs, test_case["outputs"])
                    baseline_scores.append(score)

                baseline_accuracy = statistics.mean(baseline_scores)

                # Test bootstrap optimization
                student_module = Predict(task["signature"])
                bootstrap_result = await task_bootstrap.optimize(
                    student_module, task["training_data"]
                )

                # Test optimized performance
                optimized_module = bootstrap_result.optimized_module
                bootstrap_scores = []

                for test_case in task["test_data"]:
                    result = await optimized_module.forward(**test_case["inputs"])
                    score = task["metric"](result.outputs, test_case["outputs"])
                    bootstrap_scores.append(score)

                bootstrap_accuracy = statistics.mean(bootstrap_scores)
                improvement = bootstrap_accuracy - baseline_accuracy

                success = bootstrap_accuracy >= baseline_accuracy

                task_results[task["name"]] = {
                    "baseline_accuracy": baseline_accuracy,
                    "bootstrap_accuracy": bootstrap_accuracy,
                    "improvement": improvement,
                    "success": success,
                }

                self.log_result(
                    f"{task['name']} Bootstrap Performance",
                    success,
                    f"Baseline: {baseline_accuracy:.1%}, Bootstrap: {bootstrap_accuracy:.1%} ({improvement:+.1%})",
                    {
                        "task_type": task["name"],
                        "baseline_accuracy": f"{baseline_accuracy:.1%}",
                        "bootstrap_accuracy": f"{bootstrap_accuracy:.1%}",
                        "improvement": f"{improvement:+.1%}",
                        "bootstrap_score": bootstrap_result.best_score,
                    },
                )

            except Exception as e:
                self.log_result(f"{task['name']} Bootstrap Performance", False, f"Error: {e}")
                task_results[task["name"]] = {"success": False, "error": str(e)}

        # Store results for analysis
        self.optimization_data["multi_task_results"] = task_results

        return True

    async def test_bootstrap_convergence_analysis(self):
        """Test bootstrap optimization convergence and stability."""

        print("📊 CONVERGENCE ANALYSIS: Bootstrap Optimization Stability")
        print("=" * 60)

        def stable_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            """Stable metric for convergence testing."""
            pred = str(prediction.get("answer", "")).strip()
            exp = str(expected.get("answer", "")).strip()
            return 1.0 if exp.lower() in pred.lower() else 0.0

        # Test 1: Multi-Round Bootstrap Convergence
        try:
            convergence_bootstrap = BootstrapFewShot(
                metric=stable_metric,
                max_bootstrapped_demos=3,
                max_labeled_demos=2,
                teacher_settings={"temperature": 0.2},
                student_settings={"temperature": 0.6},
                max_rounds=3,  # Multiple rounds for convergence
                metric_threshold=0.6,
            )

            student_module = Predict("question -> answer")

            convergence_training = [
                {"inputs": {"question": "What is 5 + 3?"}, "outputs": {"answer": "8"}},
                {"inputs": {"question": "What is 4 * 7?"}, "outputs": {"answer": "28"}},
                {"inputs": {"question": "What is 20 - 12?"}, "outputs": {"answer": "8"}},
            ]

            start_time = time.time()
            convergence_result = await convergence_bootstrap.optimize(
                student_module, convergence_training
            )
            end_time = time.time()

            # Analyze convergence
            rounds_completed = convergence_result.iterations
            final_score = convergence_result.best_score
            improvement = convergence_result.improvement

            success = rounds_completed > 0 and final_score >= 0 and improvement >= 0

            self.log_result(
                "Multi-Round Bootstrap Convergence",
                success,
                f"Converged in {rounds_completed} rounds with score {final_score:.2f}",
                {
                    "rounds_completed": rounds_completed,
                    "max_rounds": convergence_bootstrap.max_rounds,
                    "final_score": final_score,
                    "improvement": improvement,
                    "optimization_time": f"{end_time - start_time:.2f}s",
                    "converged_early": rounds_completed < convergence_bootstrap.max_rounds,
                },
            )

        except Exception as e:
            self.log_result("Multi-Round Bootstrap Convergence", False, f"Error: {e}")
            return False

        # Test 2: Bootstrap Stability Across Runs
        try:
            # Run bootstrap multiple times to test stability
            stability_scores = []
            stability_improvements = []

            for run_num in range(2):  # Reduced for speed
                stability_bootstrap = BootstrapFewShot(
                    metric=stable_metric,
                    max_bootstrapped_demos=2,
                    max_labeled_demos=1,
                    teacher_settings={"temperature": 0.3},
                    max_rounds=1,
                    metric_threshold=0.5,
                )

                test_module = Predict("question -> answer")
                run_result = await stability_bootstrap.optimize(test_module, convergence_training)

                stability_scores.append(run_result.best_score)
                stability_improvements.append(run_result.improvement)

            # Analyze stability
            score_variance = (
                statistics.variance(stability_scores) if len(stability_scores) > 1 else 0.0
            )
            avg_score = statistics.mean(stability_scores)
            avg_improvement = statistics.mean(stability_improvements)

            success = score_variance < 0.1 and avg_score > 0  # Low variance indicates stability

            self.log_result(
                "Bootstrap Stability Analysis",
                success,
                f"Average score: {avg_score:.2f}, Variance: {score_variance:.3f}",
                {
                    "runs_completed": len(stability_scores),
                    "average_score": avg_score,
                    "score_variance": score_variance,
                    "average_improvement": avg_improvement,
                    "scores": stability_scores,
                    "stable": score_variance < 0.1,
                },
            )

        except Exception as e:
            self.log_result("Bootstrap Stability Analysis", False, f"Error: {e}")
            return False

        return True

    def print_summary(self):
        """Print comprehensive test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        print("=" * 80)
        print("BOOTSTRAP FEW-SHOT VALIDATION SUMMARY")
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

        # Bootstrap insights
        if self.optimization_data:
            print("\n🎓 BOOTSTRAP INSIGHTS:")

            if (
                "baseline_performance" in self.optimization_data
                and "bootstrap_performance" in self.optimization_data
            ):
                baseline = self.optimization_data["baseline_performance"]
                bootstrap_perf = self.optimization_data["bootstrap_performance"]
                improvement = bootstrap_perf["improvement"]
                print(f"Performance improvement: {improvement:+.1%}")
                print(f"Bootstrap effectiveness: {'Positive' if improvement > 0 else 'Neutral'}")

            if "multi_task_results" in self.optimization_data:
                task_results = self.optimization_data["multi_task_results"]
                successful_tasks = [
                    name for name, result in task_results.items() if result.get("success")
                ]
                print(f"Successful task types: {len(successful_tasks)}/{len(task_results)}")
                for task_name in successful_tasks:
                    result = task_results[task_name]
                    print(f"  - {task_name}: {result.get('improvement', 0):+.1%} improvement")

    async def run_comprehensive_validation(self):
        """Run all bootstrap few-shot validation tests."""

        print("🎓 LogiLLM BOOTSTRAP FEW-SHOT COMPREHENSIVE VALIDATION")
        print("Testing REAL bootstrap few-shot with teacher-student example generation...")
        print("=" * 80)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ No OPENAI_API_KEY found in environment")
            return False

        # Run test suites in order
        test_suites = [
            ("Basic Bootstrap Functionality", self.test_basic_bootstrap_functionality),
            ("Bootstrap Example Generation", self.test_bootstrap_example_generation),
            ("Bootstrap Performance Improvement", self.test_bootstrap_performance_improvement),
            ("Multi-Task Bootstrapping", self.test_multi_task_bootstrapping),
            ("Bootstrap Convergence Analysis", self.test_bootstrap_convergence_analysis),
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
    """Run bootstrap few-shot validation."""
    validator = BootstrapFewShotValidator()
    success = await validator.run_comprehensive_validation()

    if success:
        print("\n🎉 BOOTSTRAP FEW-SHOT VALIDATION SUCCESSFUL!")
        print(
            "Bootstrap few-shot system is working excellently with real teacher-student example generation."
        )
    else:
        print("\n💥 BOOTSTRAP FEW-SHOT VALIDATION HAD ISSUES!")
        print("Review test results and fix bootstrap few-shot problems before proceeding.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
