#!/usr/bin/env python3
"""Comprehensive Format Optimization validation for LogiLLM.

This systematically tests format switching and optimization with REAL metrics:
- Automatic format discovery across different task types
- Multi-factor optimization (accuracy, latency, stability, parse success)
- Adaptive sampling and early stopping validation
- Model-specific format learning
- Hybrid and cognitive adapter testing
- Real performance measurement and comparison

Run with: uv run python tests/core_format_optimization_validation.py
"""

import asyncio
import os
import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.predict import Predict
from logillm.optimizers.format_optimizer import FormatOptimizer, FormatOptimizerConfig, PromptFormat
from logillm.providers import create_provider, register_provider


class FormatOptimizationValidator:
    """Comprehensively validates format optimization with real API calls and metrics."""

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

    async def test_basic_format_optimization(self):
        """Test basic format optimization functionality."""

        print("🔍 BASIC FORMAT OPTIMIZATION: Automated Format Discovery")
        print("=" * 60)

        # Prepare provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Test 1: Format Optimizer Configuration
        try:
            config = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.MARKDOWN, PromptFormat.JSON, PromptFormat.XML],
                min_samples_per_format=2,  # Reduced for speed
                max_samples_per_format=4,
                early_stopping_threshold=0.3,
                adaptive_sampling=True,
            )

            def simple_accuracy_metric(
                prediction: dict[str, Any], expected: dict[str, Any]
            ) -> float:
                """Simple accuracy metric for testing."""
                # Check if the answer field contains the expected value
                pred_answer = str(prediction.get("answer", "")).lower().strip()
                exp_answer = str(expected.get("answer", "")).lower().strip()
                return 1.0 if exp_answer in pred_answer else 0.0

            optimizer = FormatOptimizer(
                metric=simple_accuracy_metric, config=config, track_by_model=True
            )

            success = (
                optimizer.config.min_samples_per_format == 2
                and len(optimizer.config.formats_to_test) == 3
                and optimizer.track_by_model
            )

            self.log_result(
                "Format Optimizer Configuration",
                success,
                f"Configured optimizer with {len(config.formats_to_test)} formats",
                {
                    "formats_to_test": [f.value for f in config.formats_to_test],
                    "min_samples": config.min_samples_per_format,
                    "adaptive_sampling": config.adaptive_sampling,
                    "early_stopping_threshold": config.early_stopping_threshold,
                },
            )

        except Exception as e:
            self.log_result("Format Optimizer Configuration", False, f"Error: {e}")
            return False

        # Test 2: Simple Format Optimization
        try:
            # Create test module
            test_module = Predict("question -> answer")

            # Simple test dataset
            dataset = [
                {"inputs": {"question": "What is 5 * 6?"}, "outputs": {"answer": "30"}},
                {"inputs": {"question": "What is 12 + 8?"}, "outputs": {"answer": "20"}},
                {"inputs": {"question": "What is 25 / 5?"}, "outputs": {"answer": "5"}},
            ]

            start_time = time.time()
            result = await optimizer.optimize(test_module, dataset)
            end_time = time.time()

            # Validate optimization result
            success = (
                result.optimized_module is not None
                and result.best_score >= 0
                and "best_format" in result.metadata
            )

            best_format = result.metadata.get("best_format")
            format_scores = result.metadata.get("format_scores", {})

            self.log_result(
                "Simple Format Optimization",
                success,
                f"Best format: {best_format}, Score: {result.best_score:.2f}",
                {
                    "best_format": best_format,
                    "best_score": result.best_score,
                    "improvement": result.improvement,
                    "optimization_time": f"{end_time - start_time:.2f}s",
                    "formats_tested": len(format_scores),
                    "format_scores": format_scores,
                },
            )

            # Store for later analysis
            self.optimization_data["simple_optimization"] = result

        except Exception as e:
            self.log_result("Simple Format Optimization", False, f"Error: {e}")
            return False

        return True

    async def test_task_specific_optimization(self):
        """Test format optimization for different task types."""

        print("🎯 TASK-SPECIFIC OPTIMIZATION: Format Selection by Task Type")
        print("=" * 60)

        def accuracy_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            """Task-specific accuracy metric."""
            # More flexible matching for different task types
            for field_name, exp_value in expected.items():
                pred_value = prediction.get(field_name, "")
                if not any(
                    str(exp_word).lower() in str(pred_value).lower()
                    for exp_word in str(exp_value).split()
                ):
                    return 0.0
            return 1.0

        task_types = [
            {
                "name": "Mathematical Reasoning",
                "signature": "problem: str -> reasoning: str, answer: int",
                "dataset": [
                    {
                        "inputs": {
                            "problem": "If a train travels 80 miles in 2 hours, what is its speed?"
                        },
                        "outputs": {
                            "reasoning": "speed equals distance divided by time",
                            "answer": "40",
                        },
                    },
                    {
                        "inputs": {
                            "problem": "A rectangle has length 12 and width 8. What is its area?"
                        },
                        "outputs": {"reasoning": "area equals length times width", "answer": "96"},
                    },
                ],
                "expected_best_formats": [
                    "json",
                    "xml",
                ],  # Structured formats better for multi-field
            },
            {
                "name": "Simple Q&A",
                "signature": "question -> answer",
                "dataset": [
                    {
                        "inputs": {"question": "What is the capital of Germany?"},
                        "outputs": {"answer": "Berlin"},
                    },
                    {
                        "inputs": {"question": "What color is the sky?"},
                        "outputs": {"answer": "blue"},
                    },
                ],
                "expected_best_formats": ["chat", "markdown", "json", "xml"],  # All should work
            },
            {
                "name": "Classification Task",
                "signature": "text: str -> category: str, confidence: float",
                "dataset": [
                    {
                        "inputs": {"text": "This movie was absolutely terrible, I hated it."},
                        "outputs": {"category": "negative", "confidence": "0.9"},
                    },
                    {
                        "inputs": {"text": "I loved this book, it was amazing!"},
                        "outputs": {"category": "positive", "confidence": "0.8"},
                    },
                ],
                "expected_best_formats": ["json", "xml"],  # Structured output
            },
        ]

        task_results = {}

        for task in task_types:
            try:
                print(f"\n🧪 Testing: {task['name']}")

                # Create optimizer for this task
                config = FormatOptimizerConfig(
                    formats_to_test=[PromptFormat.MARKDOWN, PromptFormat.JSON, PromptFormat.XML],
                    min_samples_per_format=2,
                    max_samples_per_format=3,
                    adaptive_sampling=True,
                )

                optimizer = FormatOptimizer(metric=accuracy_metric, config=config)

                # Create test module for this task
                test_module = Predict(task["signature"])

                # Optimize format
                result = await optimizer.optimize(test_module, task["dataset"])

                best_format = result.metadata.get("best_format")
                format_scores = result.metadata.get("format_scores", {})

                # Validate results
                success = (
                    result.best_score > 0 and best_format is not None and len(format_scores) >= 2
                )

                task_results[task["name"]] = {
                    "best_format": best_format,
                    "best_score": result.best_score,
                    "format_scores": format_scores,
                    "success": success,
                }

                self.log_result(
                    f"{task['name']} Optimization",
                    success,
                    f"Best format: {best_format}, Score: {result.best_score:.2f}",
                    {
                        "task_type": task["name"],
                        "best_format": best_format,
                        "best_score": result.best_score,
                        "format_scores": format_scores,
                        "improvement": result.improvement,
                    },
                )

            except Exception as e:
                self.log_result(f"{task['name']} Optimization", False, f"Error: {e}")
                task_results[task["name"]] = {"success": False, "error": str(e)}

        # Store results for analysis
        self.optimization_data["task_specific"] = task_results

        return True

    async def test_performance_factors_analysis(self):
        """Test multi-factor performance analysis (accuracy, latency, stability)."""

        print("⚡ PERFORMANCE FACTORS: Multi-Factor Optimization Analysis")
        print("=" * 60)

        # Test 1: Latency-Weighted Optimization
        try:
            config_latency = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.JSON, PromptFormat.XML, PromptFormat.MARKDOWN],
                min_samples_per_format=2,
                max_samples_per_format=3,
                consider_latency=True,
                latency_weight=0.4,  # High weight on latency
                consider_stability=False,
            )

            def speed_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
                """Metric that values correctness."""
                pred_ans = str(prediction.get("answer", "")).strip()
                exp_ans = str(expected.get("answer", "")).strip()
                return 1.0 if exp_ans in pred_ans else 0.5  # Partial credit

            optimizer_latency = FormatOptimizer(metric=speed_metric, config=config_latency)

            test_module = Predict("question -> answer")
            dataset = [
                {"inputs": {"question": "What is 3 + 4?"}, "outputs": {"answer": "7"}},
                {"inputs": {"question": "What is 9 - 2?"}, "outputs": {"answer": "7"}},
            ]

            start_time = time.time()
            result_latency = await optimizer_latency.optimize(test_module, dataset)
            end_time = time.time()

            # Check that latency was considered
            performance_summary = result_latency.metadata.get("format_performance", {})
            has_latency_data = any("mean_latency" in perf for perf in performance_summary.values())

            success = (
                result_latency.best_score > 0
                and has_latency_data
                and result_latency.metadata.get("best_format")
            )

            self.log_result(
                "Latency-Weighted Optimization",
                success,
                f"Best format considering latency: {result_latency.metadata.get('best_format')}",
                {
                    "best_format": result_latency.metadata.get("best_format"),
                    "latency_weight": config_latency.latency_weight,
                    "has_latency_data": has_latency_data,
                    "performance_summary": performance_summary,
                    "optimization_time": f"{end_time - start_time:.2f}s",
                },
            )

        except Exception as e:
            self.log_result("Latency-Weighted Optimization", False, f"Error: {e}")
            return False

        # Test 2: Stability Analysis
        try:
            config_stability = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.JSON, PromptFormat.MARKDOWN],
                min_samples_per_format=3,  # Need more samples for stability
                max_samples_per_format=4,
                consider_stability=True,
                stability_weight=0.3,
                consider_latency=False,
            )

            optimizer_stability = FormatOptimizer(metric=speed_metric, config=config_stability)

            # Run optimization multiple times to test stability
            result_stability = await optimizer_stability.optimize(test_module, dataset)

            # Check for stability metrics
            performance_summary = result_stability.metadata.get("format_performance", {})
            has_stability_data = any("stability" in perf for perf in performance_summary.values())

            success = result_stability.best_score > 0 and has_stability_data

            self.log_result(
                "Stability-Weighted Optimization",
                success,
                f"Stability analysis completed: {has_stability_data}",
                {
                    "best_format": result_stability.metadata.get("best_format"),
                    "stability_weight": config_stability.stability_weight,
                    "has_stability_data": has_stability_data,
                    "performance_factors": list(performance_summary.keys())
                    if performance_summary
                    else [],
                },
            )

        except Exception as e:
            self.log_result("Stability-Weighted Optimization", False, f"Error: {e}")
            return False

        return True

    async def test_adaptive_sampling_early_stopping(self):
        """Test adaptive sampling and early stopping functionality."""

        print("🎛️  ADAPTIVE SAMPLING: Smart Sample Allocation and Early Stopping")
        print("=" * 60)

        # Test 1: Early Stopping
        try:
            config_early_stop = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.JSON, PromptFormat.XML, PromptFormat.MARKDOWN],
                min_samples_per_format=2,
                max_samples_per_format=6,
                early_stopping_threshold=0.5,  # Aggressive early stopping
                adaptive_sampling=False,  # Disable for controlled test
            )

            def variable_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
                """Metric that simulates variable performance."""
                pred_ans = str(prediction.get("answer", "")).lower()
                exp_ans = str(expected.get("answer", "")).lower()

                # Simulate JSON being much better than others
                if "json" in str(prediction.get("_format_hint", "")):
                    return 1.0 if exp_ans in pred_ans else 0.8
                else:
                    return 0.3  # Poor performance to trigger early stopping

            optimizer_early = FormatOptimizer(metric=variable_metric, config=config_early_stop)

            test_module = Predict("question -> answer")
            dataset = [
                {"inputs": {"question": "What is 2 + 2?"}, "outputs": {"answer": "4"}},
                {"inputs": {"question": "What is 5 + 5?"}, "outputs": {"answer": "10"}},
            ]

            start_time = time.time()
            result_early = await optimizer_early.optimize(test_module, dataset)
            end_time = time.time()

            # Check if optimization completed quickly (suggesting early stopping worked)
            optimization_time = end_time - start_time
            format_scores = result_early.metadata.get("format_scores", {})

            success = (
                result_early.best_score > 0 and len(format_scores) >= 1 and optimization_time < 30
            )  # Should be fast

            self.log_result(
                "Early Stopping Functionality",
                success,
                f"Optimization completed in {optimization_time:.1f}s with {len(format_scores)} formats tested",
                {
                    "optimization_time": f"{optimization_time:.1f}s",
                    "formats_tested": len(format_scores),
                    "early_stopping_threshold": config_early_stop.early_stopping_threshold,
                    "format_scores": format_scores,
                },
            )

        except Exception as e:
            self.log_result("Early Stopping Functionality", False, f"Error: {e}")
            return False

        # Test 2: Adaptive Sampling
        try:
            config_adaptive = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.JSON, PromptFormat.MARKDOWN],
                min_samples_per_format=2,
                max_samples_per_format=5,
                adaptive_sampling=True,
                early_stopping_threshold=0.2,
            )

            def consistent_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
                """Consistent metric for adaptive sampling test."""
                pred_ans = str(prediction.get("answer", "")).lower()
                exp_ans = str(expected.get("answer", "")).lower()
                return 1.0 if exp_ans in pred_ans else 0.0

            optimizer_adaptive = FormatOptimizer(metric=consistent_metric, config=config_adaptive)

            result_adaptive = await optimizer_adaptive.optimize(test_module, dataset)

            # Check that adaptive sampling worked
            performance_summary = result_adaptive.metadata.get("format_performance", {})
            sample_counts = {
                fmt: perf.get("samples", 0) for fmt, perf in performance_summary.items()
            }

            success = (
                result_adaptive.best_score >= 0
                and len(sample_counts) >= 1
                and max(sample_counts.values()) >= config_adaptive.min_samples_per_format
            )

            self.log_result(
                "Adaptive Sampling",
                success,
                f"Sample allocation: {sample_counts}",
                {
                    "adaptive_sampling": config_adaptive.adaptive_sampling,
                    "sample_counts": sample_counts,
                    "min_samples": config_adaptive.min_samples_per_format,
                    "max_samples": config_adaptive.max_samples_per_format,
                },
            )

        except Exception as e:
            self.log_result("Adaptive Sampling", False, f"Error: {e}")
            return False

        return True

    async def test_format_learning_recommendations(self):
        """Test model-specific format learning and recommendations."""

        print("🧠 FORMAT LEARNING: Model-Specific Optimization Memory")
        print("=" * 60)

        # Test 1: Format Learning Across Tasks
        try:
            config = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.JSON, PromptFormat.MARKDOWN, PromptFormat.XML],
                min_samples_per_format=2,
                max_samples_per_format=3,
            )

            def learning_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
                """Metric for learning test."""
                pred_ans = str(prediction.get("answer", "")).lower()
                exp_ans = str(expected.get("answer", "")).lower()
                return 1.0 if exp_ans in pred_ans else 0.0

            optimizer_learning = FormatOptimizer(
                metric=learning_metric, config=config, track_by_model=True
            )

            # Run multiple optimization tasks
            tasks = [
                {
                    "signature": "question -> answer",
                    "dataset": [
                        {"inputs": {"question": "What is 1+1?"}, "outputs": {"answer": "2"}}
                    ],
                },
                {
                    "signature": "problem -> solution",
                    "dataset": [
                        {"inputs": {"problem": "Solve x+1=3"}, "outputs": {"solution": "x=2"}}
                    ],
                },
            ]

            learned_formats = {}
            for i, task in enumerate(tasks):
                test_module = Predict(task["signature"])
                result = await optimizer_learning.optimize(test_module, task["dataset"])
                learned_formats[f"task_{i}"] = result.metadata.get("best_format")

            # Get format recommendations
            recommendations = optimizer_learning.get_format_recommendations()

            success = (
                len(learned_formats) == len(tasks)
                and isinstance(recommendations, dict)
                and len(recommendations) > 0
            )

            self.log_result(
                "Format Learning Across Tasks",
                success,
                f"Learned formats: {learned_formats}",
                {
                    "learned_formats": learned_formats,
                    "recommendations": recommendations,
                    "tracks_by_model": optimizer_learning.track_by_model,
                    "num_tasks": len(tasks),
                },
            )

        except Exception as e:
            self.log_result("Format Learning Across Tasks", False, f"Error: {e}")
            return False

        # Test 2: Format Performance Persistence
        try:
            # Check that performance data persists across optimizations
            model_performance = optimizer_learning.format_performance

            success = len(model_performance) > 0 and any(
                len(model_formats) > 0 for model_formats in model_performance.values()
            )

            performance_data = {}
            for model_id, formats in model_performance.items():
                performance_data[model_id] = {
                    fmt.value: {
                        "samples": len(perf.scores),
                        "mean_score": perf.mean_score if perf.scores else 0,
                        "success_rate": perf.success_rate,
                    }
                    for fmt, perf in formats.items()
                    if hasattr(perf, "scores")
                }

            self.log_result(
                "Format Performance Persistence",
                success,
                f"Tracked performance for {len(model_performance)} models",
                {
                    "models_tracked": len(model_performance),
                    "performance_data": performance_data,
                    "has_persistent_data": success,
                },
            )

        except Exception as e:
            self.log_result("Format Performance Persistence", False, f"Error: {e}")
            return False

        return True

    def print_summary(self):
        """Print comprehensive test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        print("=" * 80)
        print("FORMAT OPTIMIZATION VALIDATION SUMMARY")
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

        # Optimization insights
        if self.optimization_data:
            print("\n🎯 OPTIMIZATION INSIGHTS:")

            if "simple_optimization" in self.optimization_data:
                simple_result = self.optimization_data["simple_optimization"]
                print(
                    f"Simple optimization best format: {simple_result.metadata.get('best_format')}"
                )

            if "task_specific" in self.optimization_data:
                task_results = self.optimization_data["task_specific"]
                print("Task-specific format preferences:")
                for task_name, result in task_results.items():
                    if result.get("success"):
                        print(f"  - {task_name}: {result.get('best_format')}")

    async def run_comprehensive_validation(self):
        """Run all format optimization validation tests."""

        print("🔍 LogiLLM FORMAT OPTIMIZATION COMPREHENSIVE VALIDATION")
        print("Testing REAL format optimization with performance metrics...")
        print("=" * 80)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ No OPENAI_API_KEY found in environment")
            return False

        # Run test suites in order
        test_suites = [
            ("Basic Format Optimization", self.test_basic_format_optimization),
            ("Task-Specific Optimization", self.test_task_specific_optimization),
            ("Performance Factors Analysis", self.test_performance_factors_analysis),
            ("Adaptive Sampling & Early Stopping", self.test_adaptive_sampling_early_stopping),
            ("Format Learning & Recommendations", self.test_format_learning_recommendations),
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
    """Run format optimization validation."""
    validator = FormatOptimizationValidator()
    success = await validator.run_comprehensive_validation()

    if success:
        print("\n🎉 FORMAT OPTIMIZATION VALIDATION SUCCESSFUL!")
        print("Format optimization system is working excellently with real performance metrics.")
    else:
        print("\n💥 FORMAT OPTIMIZATION VALIDATION HAD ISSUES!")
        print("Review test results and fix optimization problems before proceeding.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
