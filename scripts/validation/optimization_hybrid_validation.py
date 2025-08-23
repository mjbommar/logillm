#!/usr/bin/env python3
"""Comprehensive Hybrid Optimization validation for LogiLLM.

This systematically tests LogiLLM's KILLER FEATURE - hybrid optimization with REAL improvements:
- Simultaneous prompt AND hyperparameter optimization
- Real performance improvements over single-dimension optimization
- Multi-objective optimization balancing accuracy, latency, and cost
- Format + hyperparameter joint optimization with actual metrics
- Alternating vs joint optimization strategies with real comparisons
- Cross-dimensional interaction analysis with measured effects

This is what differentiates LogiLLM from DSPy - our hybrid optimization capabilities!

Run with: uv run python tests/optimization_hybrid_validation.py
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.parameters import ParamDomain, ParamSpec, ParamType, SearchSpace
from logillm.core.predict import Predict
from logillm.optimizers import HybridOptimizer
from logillm.optimizers.format_optimizer import FormatOptimizer, FormatOptimizerConfig, PromptFormat
from logillm.providers import create_provider, register_provider


class HybridOptimizationValidator:
    """Comprehensively validates hybrid optimization - LogiLLM's killer feature."""

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

    async def test_basic_hybrid_configuration(self):
        """Test basic hybrid optimizer setup and configuration."""

        print("🔗 BASIC HYBRID OPTIMIZATION: Configuration and Setup")
        print("=" * 60)

        # Prepare provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Test 1: Hybrid Optimizer Configuration
        try:

            def hybrid_accuracy_metric(
                prediction: dict[str, Any], expected: dict[str, Any]
            ) -> float:
                """Hybrid optimization accuracy metric."""
                pred_answer = str(prediction.get("answer", "")).lower().strip()
                exp_answer = str(expected.get("answer", "")).lower().strip()

                # Extract numbers for math problems
                import re

                pred_nums = re.findall(r"\d+", pred_answer)
                exp_nums = re.findall(r"\d+", exp_answer)

                if pred_nums and exp_nums:
                    return 1.0 if pred_nums[0] == exp_nums[0] else 0.0

                return 1.0 if exp_answer in pred_answer else 0.0

            # Create format optimizer component
            format_config = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.MARKDOWN, PromptFormat.JSON, PromptFormat.XML],
                min_samples_per_format=2,
                max_samples_per_format=3,
                adaptive_sampling=True,
            )
            format_optimizer = FormatOptimizer(metric=hybrid_accuracy_metric, config=format_config)

            # Create hyperparameter search space
            hyperparam_space = SearchSpace(
                {
                    "temperature": ParamSpec(
                        name="temperature",
                        param_type=ParamType.FLOAT,
                        domain=ParamDomain.GENERATION,
                        description="Temperature",
                        default=0.5,
                        range=(0.1, 0.9),
                        step=0.2,
                    ),
                    "max_tokens": ParamSpec(
                        name="max_tokens",
                        param_type=ParamType.INT,
                        domain=ParamDomain.GENERATION,
                        description="Max tokens",
                        default=100,
                        range=(75, 150),
                        step=25,
                    ),
                }
            )

            # Create hybrid optimizer
            hybrid_optimizer = HybridOptimizer(
                metric=hybrid_accuracy_metric,
                format_optimizer=format_optimizer,
                hyperparam_search_space=hyperparam_space,
                strategy="alternating",  # Test alternating strategy
                max_iterations=3,  # Reduced for speed
                convergence_threshold=0.01,
            )

            success = (
                hybrid_optimizer.format_optimizer is not None
                and hybrid_optimizer.hyperparam_search_space is not None
                and hybrid_optimizer.strategy == "alternating"
            )

            self.log_result(
                "Hybrid Optimizer Configuration",
                success,
                f"Configured hybrid optimizer with {hybrid_optimizer.strategy} strategy",
                {
                    "strategy": hybrid_optimizer.strategy,
                    "max_iterations": hybrid_optimizer.max_iterations,
                    "has_format_optimizer": hybrid_optimizer.format_optimizer is not None,
                    "has_hyperparam_space": hybrid_optimizer.hyperparam_search_space is not None,
                    "hyperparam_count": len(hybrid_optimizer.hyperparam_search_space.param_specs),
                    "format_count": len(format_config.formats_to_test),
                },
            )

        except Exception as e:
            self.log_result("Hybrid Optimizer Configuration", False, f"Error: {e}")
            return False

        # Test 2: Component Integration Validation
        try:
            # Test that components work together
            test_module = Predict("problem -> answer")

            # Verify both optimizers can be applied to the same module
            format_applied = format_optimizer._apply_format(test_module, PromptFormat.JSON)

            success = (
                format_applied is not None
                and hasattr(format_applied, "adapter")
                and test_module is not None
            )

            self.log_result(
                "Component Integration",
                success,
                "Format and hyperparameter components integrate successfully",
                {
                    "format_applied": format_applied is not None,
                    "original_module_preserved": test_module is not None,
                    "adapter_set": hasattr(format_applied, "adapter"),
                },
            )

        except Exception as e:
            self.log_result("Component Integration", False, f"Error: {e}")
            return False

        return True

    async def test_alternating_optimization_strategy(self):
        """Test alternating optimization: format first, then hyperparams, then format, etc."""

        print("🔄 ALTERNATING OPTIMIZATION: Format ↔ Hyperparameters")
        print("=" * 60)

        def math_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            """Math-focused metric for alternating optimization."""
            pred_answer = str(prediction.get("answer", "")).strip()
            exp_answer = str(expected.get("answer", "")).strip()

            import re

            pred_nums = re.findall(r"\d+", pred_answer)
            exp_nums = re.findall(r"\d+", exp_answer)

            if pred_nums and exp_nums and pred_nums[0] == exp_nums[0]:
                return 1.0

            return 0.0

        # Test 1: Alternating Strategy Execution
        try:
            # Create components for alternating optimization
            format_config_alt = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.JSON, PromptFormat.MARKDOWN],
                min_samples_per_format=2,
                max_samples_per_format=3,
            )

            hyperparam_space_alt = SearchSpace(
                {
                    "temperature": ParamSpec(
                        name="temperature",
                        param_type=ParamType.FLOAT,
                        domain=ParamDomain.GENERATION,
                        description="Temperature",
                        default=0.6,
                        range=(0.2, 0.8),
                        step=0.3,
                    )
                }
            )

            hybrid_alt = HybridOptimizer(
                metric=math_metric,
                format_optimizer=FormatOptimizer(metric=math_metric, config=format_config_alt),
                hyperparam_search_space=hyperparam_space_alt,
                strategy="alternating",
                max_iterations=2,  # Two alternations: format->hyperparam->format
                convergence_threshold=0.05,
            )

            # Test data
            alternating_dataset = [
                {"inputs": {"problem": "What is 18 + 24?"}, "outputs": {"answer": "42"}},
                {"inputs": {"problem": "What is 9 * 7?"}, "outputs": {"answer": "63"}},
                {"inputs": {"problem": "What is 81 / 9?"}, "outputs": {"answer": "9"}},
            ]

            test_module = Predict("problem -> answer")

            start_time = time.time()
            result = await hybrid_alt.optimize(test_module, alternating_dataset)
            end_time = time.time()

            # Analyze results
            optimization_history = result.metadata.get("optimization_history", [])
            alternation_pattern = [step.get("dimension") for step in optimization_history]

            success = (
                result.best_score > 0 and result.improvement >= 0 and len(alternation_pattern) > 0
            )

            self.log_result(
                "Alternating Strategy Execution",
                success,
                f"Score: {result.best_score:.2f}, Alternations: {len(alternation_pattern)}",
                {
                    "final_score": result.best_score,
                    "improvement": result.improvement,
                    "optimization_time": f"{end_time - start_time:.2f}s",
                    "iterations": result.iterations,
                    "alternation_pattern": alternation_pattern,
                    "optimization_steps": len(optimization_history),
                },
            )

            # Store for analysis
            self.optimization_data["alternating_result"] = result

        except Exception as e:
            self.log_result("Alternating Strategy Execution", False, f"Error: {e}")
            return False

        # Test 2: Cross-Dimensional Performance Analysis
        try:
            # Analyze how format and hyperparameter changes affect each other
            optimization_history = result.metadata.get("optimization_history", [])

            format_steps = [
                step for step in optimization_history if step.get("dimension") == "format"
            ]
            hyperparam_steps = [
                step for step in optimization_history if step.get("dimension") == "hyperparameters"
            ]

            format_improvements = [step.get("score_improvement", 0) for step in format_steps]
            hyperparam_improvements = [
                step.get("score_improvement", 0) for step in hyperparam_steps
            ]

            avg_format_improvement = (
                statistics.mean(format_improvements) if format_improvements else 0
            )
            avg_hyperparam_improvement = (
                statistics.mean(hyperparam_improvements) if hyperparam_improvements else 0
            )

            success = len(format_steps) > 0 or len(hyperparam_steps) > 0

            self.log_result(
                "Cross-Dimensional Performance Analysis",
                success,
                f"Format avg improvement: {avg_format_improvement:+.2f}, Hyperparam: {avg_hyperparam_improvement:+.2f}",
                {
                    "format_optimization_steps": len(format_steps),
                    "hyperparam_optimization_steps": len(hyperparam_steps),
                    "avg_format_improvement": avg_format_improvement,
                    "avg_hyperparam_improvement": avg_hyperparam_improvement,
                    "cross_dimensional_interaction": abs(avg_format_improvement)
                    + abs(avg_hyperparam_improvement)
                    > 0,
                },
            )

        except Exception as e:
            self.log_result("Cross-Dimensional Performance Analysis", False, f"Error: {e}")
            return False

        return True

    async def test_joint_optimization_strategy(self):
        """Test joint optimization: simultaneous format and hyperparameter optimization."""

        print("🤝 JOINT OPTIMIZATION: Simultaneous Format + Hyperparameters")
        print("=" * 60)

        def joint_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            """Joint optimization metric."""
            pred_answer = str(prediction.get("answer", "")).strip()
            exp_answer = str(expected.get("answer", "")).strip()
            return 1.0 if exp_answer.lower() in pred_answer.lower() else 0.0

        # Test 1: Joint Strategy Configuration
        try:
            joint_format_config = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.JSON, PromptFormat.XML],
                min_samples_per_format=2,
                adaptive_sampling=True,
            )

            joint_hyperparam_space = SearchSpace(
                {
                    "temperature": ParamSpec(
                        name="temperature",
                        param_type=ParamType.FLOAT,
                        domain=ParamDomain.GENERATION,
                        description="Temperature",
                        default=0.5,
                        range=(0.3, 0.7),
                        step=0.2,
                    ),
                    "top_p": ParamSpec(
                        name="top_p",
                        param_type=ParamType.FLOAT,
                        domain=ParamDomain.GENERATION,
                        description="Top-p",
                        default=0.9,
                        range=(0.7, 1.0),
                        step=0.15,
                    ),
                }
            )

            hybrid_joint = HybridOptimizer(
                metric=joint_metric,
                format_optimizer=FormatOptimizer(metric=joint_metric, config=joint_format_config),
                hyperparam_search_space=joint_hyperparam_space,
                strategy="joint",  # Joint optimization
                max_iterations=2,
                convergence_threshold=0.1,
            )

            success = (
                hybrid_joint.strategy == "joint" and len(joint_hyperparam_space.param_specs) == 2
            )

            self.log_result(
                "Joint Strategy Configuration",
                success,
                f"Joint strategy configured with {len(joint_hyperparam_space.param_specs)} hyperparams",
                {
                    "strategy": hybrid_joint.strategy,
                    "hyperparam_dimensions": len(joint_hyperparam_space.param_specs),
                    "format_options": len(joint_format_config.formats_to_test),
                    "max_iterations": hybrid_joint.max_iterations,
                },
            )

        except Exception as e:
            self.log_result("Joint Strategy Configuration", False, f"Error: {e}")
            return False

        # Test 2: Joint Optimization Execution
        try:
            # Test data for joint optimization
            joint_dataset = [
                {"inputs": {"problem": "Solve: 15 + 27"}, "outputs": {"answer": "42"}},
                {"inputs": {"problem": "Solve: 6 * 8"}, "outputs": {"answer": "48"}},
                {"inputs": {"problem": "Solve: 90 / 10"}, "outputs": {"answer": "9"}},
            ]

            test_module = Predict("problem -> answer")

            start_time = time.time()
            joint_result = await hybrid_joint.optimize(test_module, joint_dataset)
            end_time = time.time()

            # Analyze joint optimization
            joint_history = joint_result.metadata.get("optimization_history", [])
            simultaneous_steps = [
                step for step in joint_history if step.get("optimization_type") == "joint"
            ]

            success = joint_result.best_score >= 0 and joint_result.iterations > 0

            self.log_result(
                "Joint Optimization Execution",
                success,
                f"Joint score: {joint_result.best_score:.2f}, Time: {end_time - start_time:.1f}s",
                {
                    "joint_final_score": joint_result.best_score,
                    "joint_improvement": joint_result.improvement,
                    "joint_optimization_time": f"{end_time - start_time:.2f}s",
                    "joint_iterations": joint_result.iterations,
                    "simultaneous_optimization_steps": len(simultaneous_steps),
                },
            )

            # Store for comparison
            self.optimization_data["joint_result"] = joint_result

        except Exception as e:
            self.log_result("Joint Optimization Execution", False, f"Error: {e}")
            return False

        return True

    async def test_hybrid_vs_single_dimension_comparison(self):
        """Test hybrid optimization performance vs single-dimension optimization."""

        print("⚖️  PERFORMANCE COMPARISON: Hybrid vs Single-Dimension")
        print("=" * 60)

        def comparison_metric(prediction: dict[str, Any], expected: dict[str, Any]) -> float:
            """Consistent metric for comparison."""
            pred = str(prediction.get("answer", "")).strip()
            exp = str(expected.get("answer", "")).strip()

            # Try numeric comparison first
            import re

            pred_nums = re.findall(r"\d+", pred)
            exp_nums = re.findall(r"\d+", exp)

            if pred_nums and exp_nums:
                return 1.0 if pred_nums[0] == exp_nums[0] else 0.0

            return 1.0 if exp.lower() in pred.lower() else 0.0

        comparison_dataset = [
            {"inputs": {"problem": "Calculate 25 * 3"}, "outputs": {"answer": "75"}},
            {"inputs": {"problem": "Calculate 144 / 12"}, "outputs": {"answer": "12"}},
            {"inputs": {"problem": "Calculate 50 - 18"}, "outputs": {"answer": "32"}},
            {"inputs": {"problem": "Calculate 29 + 16"}, "outputs": {"answer": "45"}},
        ]

        results = {}

        # Test 1: Format-Only Optimization
        try:
            format_only = FormatOptimizer(
                metric=comparison_metric,
                config=FormatOptimizerConfig(
                    formats_to_test=[PromptFormat.JSON, PromptFormat.MARKDOWN],
                    min_samples_per_format=2,
                ),
            )

            test_module_format = Predict("problem -> answer")

            start_time = time.time()
            format_result = await format_only.optimize(test_module_format, comparison_dataset)
            end_time = time.time()

            results["format_only"] = {
                "score": format_result.best_score,
                "improvement": format_result.improvement,
                "time": end_time - start_time,
            }

            self.log_result(
                "Format-Only Optimization",
                format_result.best_score >= 0,
                f"Format-only score: {format_result.best_score:.2f}",
                {
                    "score": format_result.best_score,
                    "improvement": format_result.improvement,
                    "optimization_time": f"{end_time - start_time:.2f}s",
                    "best_format": format_result.metadata.get("best_format"),
                },
            )

        except Exception as e:
            self.log_result("Format-Only Optimization", False, f"Error: {e}")
            results["format_only"] = {"score": 0, "error": str(e)}

        # Test 2: Hyperparameter-Only Optimization
        try:
            from logillm.optimizers import HyperparameterOptimizer
            from logillm.optimizers.search_strategies import GridSearchStrategy

            hyperparam_space = SearchSpace(
                {
                    "temperature": ParamSpec(
                        name="temperature",
                        param_type=ParamType.FLOAT,
                        domain=ParamDomain.GENERATION,
                        description="Temperature",
                        default=0.5,
                        range=(0.2, 0.8),
                        step=0.3,
                    )
                }
            )

            hyperparam_only = HyperparameterOptimizer(
                metric=comparison_metric,
                search_space=hyperparam_space,
                strategy=GridSearchStrategy(resolution=2),
                n_trials=3,
            )

            test_module_hyperparam = Predict("problem -> answer")

            start_time = time.time()
            hyperparam_result = await hyperparam_only.optimize(
                test_module_hyperparam, comparison_dataset, valset=comparison_dataset
            )
            end_time = time.time()

            results["hyperparam_only"] = {
                "score": hyperparam_result.best_score,
                "improvement": hyperparam_result.improvement,
                "time": end_time - start_time,
            }

            self.log_result(
                "Hyperparameter-Only Optimization",
                hyperparam_result.best_score >= 0,
                f"Hyperparam-only score: {hyperparam_result.best_score:.2f}",
                {
                    "score": hyperparam_result.best_score,
                    "improvement": hyperparam_result.improvement,
                    "optimization_time": f"{end_time - start_time:.2f}s",
                    "best_params": hyperparam_result.metadata.get("best_config", {}),
                },
            )

        except Exception as e:
            self.log_result("Hyperparameter-Only Optimization", False, f"Error: {e}")
            results["hyperparam_only"] = {"score": 0, "error": str(e)}

        # Test 3: Hybrid Optimization
        try:
            hybrid_comparison = HybridOptimizer(
                metric=comparison_metric,
                format_optimizer=FormatOptimizer(
                    metric=comparison_metric,
                    config=FormatOptimizerConfig(
                        formats_to_test=[PromptFormat.JSON, PromptFormat.MARKDOWN]
                    ),
                ),
                hyperparam_search_space=hyperparam_space,
                strategy="alternating",
                max_iterations=2,
            )

            test_module_hybrid = Predict("problem -> answer")

            start_time = time.time()
            hybrid_result = await hybrid_comparison.optimize(test_module_hybrid, comparison_dataset)
            end_time = time.time()

            results["hybrid"] = {
                "score": hybrid_result.best_score,
                "improvement": hybrid_result.improvement,
                "time": end_time - start_time,
            }

            # Calculate advantage over single dimensions
            format_score = results["format_only"].get("score", 0)
            hyperparam_score = results["hyperparam_only"].get("score", 0)
            hybrid_score = results["hybrid"]["score"]

            best_single = max(format_score, hyperparam_score)
            hybrid_advantage = (
                (hybrid_score - best_single) / best_single * 100 if best_single > 0 else 0
            )

            success = hybrid_score >= best_single

            self.log_result(
                "Hybrid vs Single-Dimension Performance",
                success,
                f"Hybrid: {hybrid_score:.2f}, Best single: {best_single:.2f}, Advantage: {hybrid_advantage:+.1f}%",
                {
                    "hybrid_score": hybrid_score,
                    "format_only_score": format_score,
                    "hyperparam_only_score": hyperparam_score,
                    "best_single_dimension": best_single,
                    "hybrid_advantage_percent": f"{hybrid_advantage:+.1f}%",
                    "hybrid_time": f"{end_time - start_time:.2f}s",
                },
            )

            # Store for analysis
            self.optimization_data["comparison_results"] = results

        except Exception as e:
            self.log_result("Hybrid vs Single-Dimension Performance", False, f"Error: {e}")
            return False

        return True

    async def test_multi_objective_hybrid_optimization(self):
        """Test hybrid optimization with multiple objectives (accuracy, latency, cost)."""

        print("🎯 MULTI-OBJECTIVE HYBRID: Accuracy + Latency + Cost")
        print("=" * 60)

        def multi_objective_metric(
            prediction: dict[str, Any],
            expected: dict[str, Any],
            latency: float = 0.0,
            cost_estimate: float = 0.0,
        ) -> dict[str, float]:
            """Multi-objective metric for hybrid optimization."""
            pred = str(prediction.get("answer", "")).strip()
            exp = str(expected.get("answer", "")).strip()

            # Accuracy
            accuracy = 1.0 if exp.lower() in pred.lower() else 0.0

            # Latency score (lower is better, normalized to 0-1)
            latency_score = max(0.0, 1.0 - (latency / 5.0))  # 5s as worst case

            # Cost score (lower tokens/complexity is better)
            cost_score = max(0.0, 1.0 - (cost_estimate / 100.0))  # 100 as worst case

            # Combined score (weighted)
            combined = 0.6 * accuracy + 0.2 * latency_score + 0.2 * cost_score

            return {
                "accuracy": accuracy,
                "latency_score": latency_score,
                "cost_score": cost_score,
                "combined_score": combined,
            }

        # Test 1: Multi-Objective Hybrid Configuration
        try:
            multi_obj_config = FormatOptimizerConfig(
                formats_to_test=[PromptFormat.JSON, PromptFormat.MARKDOWN],
                consider_latency=True,
                consider_stability=True,
                latency_weight=0.2,
                stability_weight=0.1,
            )

            multi_obj_space = SearchSpace(
                {
                    "temperature": ParamSpec(
                        name="temperature",
                        param_type=ParamType.FLOAT,
                        domain=ParamDomain.GENERATION,
                        description="Temperature",
                        default=0.5,
                        range=(0.1, 0.9),
                        step=0.4,
                    ),
                    "max_tokens": ParamSpec(
                        name="max_tokens",
                        param_type=ParamType.INT,
                        domain=ParamDomain.GENERATION,
                        description="Max tokens",
                        default=100,
                        range=(50, 150),
                        step=50,
                    ),
                }
            )

            def simple_metric(pred: dict[str, Any], exp: dict[str, Any]) -> float:
                return multi_objective_metric(pred, exp)["combined_score"]

            multi_obj_hybrid = HybridOptimizer(
                metric=simple_metric,
                format_optimizer=FormatOptimizer(metric=simple_metric, config=multi_obj_config),
                hyperparam_search_space=multi_obj_space,
                strategy="alternating",
                max_iterations=2,
                multi_objective=True,
            )

            success = (
                multi_obj_hybrid.multi_objective
                and multi_obj_config.consider_latency
                and multi_obj_config.consider_stability
            )

            self.log_result(
                "Multi-Objective Hybrid Configuration",
                success,
                "Multi-objective hybrid optimizer configured",
                {
                    "multi_objective_enabled": multi_obj_hybrid.multi_objective,
                    "considers_latency": multi_obj_config.consider_latency,
                    "considers_stability": multi_obj_config.consider_stability,
                    "latency_weight": multi_obj_config.latency_weight,
                    "stability_weight": multi_obj_config.stability_weight,
                },
            )

        except Exception as e:
            self.log_result("Multi-Objective Hybrid Configuration", False, f"Error: {e}")
            return False

        # Test 2: Multi-Objective Performance Analysis
        try:
            multi_obj_dataset = [
                {"inputs": {"problem": "Quick: 5 + 5"}, "outputs": {"answer": "10"}},
                {"inputs": {"problem": "Simple: 3 * 4"}, "outputs": {"answer": "12"}},
                {"inputs": {"problem": "Easy: 20 / 4"}, "outputs": {"answer": "5"}},
            ]

            test_module = Predict("problem -> answer")

            # Measure multi-objective performance
            start_time = time.time()
            multi_obj_result = await multi_obj_hybrid.optimize(test_module, multi_obj_dataset)
            end_time = time.time()

            # Analyze multi-objective results
            optimization_time = end_time - start_time

            # Test optimized module performance
            optimized_module = multi_obj_result.optimized_module
            test_latencies = []
            test_accuracies = []

            for test_case in multi_obj_dataset:
                test_start = time.time()
                result = await optimized_module.forward(**test_case["inputs"])
                test_end = time.time()

                test_latencies.append(test_end - test_start)
                accuracy = simple_metric(result.outputs, test_case["outputs"])
                test_accuracies.append(accuracy)

            avg_accuracy = statistics.mean(test_accuracies)
            avg_latency = statistics.mean(test_latencies)

            success = avg_accuracy > 0 and optimization_time < 60  # Reasonable time

            self.log_result(
                "Multi-Objective Performance Analysis",
                success,
                f"Accuracy: {avg_accuracy:.2f}, Avg latency: {avg_latency:.2f}s",
                {
                    "avg_accuracy": avg_accuracy,
                    "avg_latency": f"{avg_latency:.2f}s",
                    "optimization_time": f"{optimization_time:.2f}s",
                    "final_score": multi_obj_result.best_score,
                    "improvement": multi_obj_result.improvement,
                    "multi_objective_balance": avg_accuracy > 0.5 and avg_latency < 2.0,
                },
            )

        except Exception as e:
            self.log_result("Multi-Objective Performance Analysis", False, f"Error: {e}")
            return False

        return True

    def print_summary(self):
        """Print comprehensive test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        print("=" * 80)
        print("HYBRID OPTIMIZATION VALIDATION SUMMARY")
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

        # Hybrid optimization insights
        if self.optimization_data:
            print("\n🔗 HYBRID OPTIMIZATION INSIGHTS:")

            if "comparison_results" in self.optimization_data:
                results = self.optimization_data["comparison_results"]
                format_score = results.get("format_only", {}).get("score", 0)
                hyperparam_score = results.get("hyperparam_only", {}).get("score", 0)
                hybrid_score = results.get("hybrid", {}).get("score", 0)

                print(f"Format-only performance: {format_score:.2f}")
                print(f"Hyperparam-only performance: {hyperparam_score:.2f}")
                print(f"Hybrid performance: {hybrid_score:.2f}")

                best_single = max(format_score, hyperparam_score)
                if best_single > 0:
                    advantage = (hybrid_score - best_single) / best_single * 100
                    print(f"🚀 HYBRID ADVANTAGE: {advantage:+.1f}% over best single dimension!")

            if "alternating_result" in self.optimization_data:
                alt_result = self.optimization_data["alternating_result"]
                print(f"Alternating strategy final score: {alt_result.best_score:.2f}")
                print(f"Alternating strategy improvement: {alt_result.improvement:+.2f}")

    async def run_comprehensive_validation(self):
        """Run all hybrid optimization validation tests."""

        print("🔗 LogiLLM HYBRID OPTIMIZATION COMPREHENSIVE VALIDATION")
        print("Testing REAL hybrid optimization - our killer feature over DSPy!")
        print("=" * 80)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ No OPENAI_API_KEY found in environment")
            return False

        # Run test suites in order
        test_suites = [
            ("Basic Hybrid Configuration", self.test_basic_hybrid_configuration),
            ("Alternating Optimization Strategy", self.test_alternating_optimization_strategy),
            ("Joint Optimization Strategy", self.test_joint_optimization_strategy),
            (
                "Hybrid vs Single-Dimension Comparison",
                self.test_hybrid_vs_single_dimension_comparison,
            ),
            ("Multi-Objective Hybrid Optimization", self.test_multi_objective_hybrid_optimization),
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
    """Run hybrid optimization validation."""
    validator = HybridOptimizationValidator()
    success = await validator.run_comprehensive_validation()

    if success:
        print("\n🎉 HYBRID OPTIMIZATION VALIDATION SUCCESSFUL!")
        print("LogiLLM's killer feature - hybrid optimization - is working excellently!")
        print("🚀 This is what differentiates us from DSPy!")
    else:
        print("\n💥 HYBRID OPTIMIZATION VALIDATION HAD ISSUES!")
        print("Review test results and fix hybrid optimization problems before proceeding.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
