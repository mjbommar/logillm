#!/usr/bin/env python3
"""Multi-model validation for LogiLLM.

Tests LogiLLM against the latest OpenAI models: gpt-4.1 and gpt-5
to ensure compatibility and performance across model generations.

Run with: uv run python tests/multi_model_validation.py
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.predict import ChainOfThought, Predict
from logillm.providers import create_provider, register_provider


class MultiModelValidator:
    """Validates LogiLLM across multiple OpenAI models."""

    def __init__(self):
        """Initialize validator."""
        self.results = {}
        self.models = ["gpt-4.1", "gpt-5"]  # Latest models

    def log_result(
        self,
        model: str,
        test_name: str,
        success: bool,
        details: str = "",
        metrics: Dict[str, Any] = None,
    ):
        """Log test result for a specific model."""
        if model not in self.results:
            self.results[model] = []

        status = "✅" if success else "❌"
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "metrics": metrics or {},
        }
        self.results[model].append(result)
        print(f"{status} {model} - {test_name}: {details}")
        if metrics:
            for key, value in metrics.items():
                print(f"   📊 {key}: {value}")
        print()

    def print_summary(self):
        """Print comprehensive test summary."""
        print("=" * 80)
        print("MULTI-MODEL VALIDATION SUMMARY")
        print("=" * 80)

        overall_stats = {}
        for model in self.models:
            if model in self.results:
                total = len(self.results[model])
                passed = sum(1 for r in self.results[model] if r["success"])
                success_rate = (passed / total * 100) if total > 0 else 0

                overall_stats[model] = {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "success_rate": success_rate,
                }

                print(f"\n📊 {model.upper()}")
                print(f"   Total tests: {total}")
                print(f"   Passed: {passed}")
                print(f"   Failed: {total - passed}")
                print(f"   Success rate: {success_rate:.1f}%")

        # Overall summary
        if overall_stats:
            total_tests = sum(stats["total"] for stats in overall_stats.values())
            total_passed = sum(stats["passed"] for stats in overall_stats.values())
            overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

            print("\n🎯 OVERALL RESULTS")
            print(f"   Models tested: {len(overall_stats)}")
            print(f"   Total tests: {total_tests}")
            print(f"   Total passed: {total_passed}")
            print(f"   Overall success rate: {overall_success_rate:.1f}%")

        print("=" * 80)

        # Show failures
        for model, results in self.results.items():
            failures = [r for r in results if not r["success"]]
            if failures:
                print(f"\n❌ {model.upper()} FAILURES:")
                for result in failures:
                    print(f"  - {result['test']}: {result['details']}")

    async def test_model_basic_functionality(self, model: str):
        """Test basic functionality for a specific model."""
        print(f"🧪 Testing {model.upper()} - Basic Functionality")
        print("-" * 60)

        try:
            # Test 1: Provider Creation
            provider = create_provider("openai", model=model)
            self.log_result(
                model,
                "Provider Creation",
                True,
                f"Successfully created {model} provider",
                {"model": model, "provider_type": "openai"},
            )

            # Test 2: Basic Completion
            start_time = time.time()
            messages = [{"role": "user", "content": "Say exactly: MODEL_TEST_SUCCESS"}]
            completion = await provider.complete(messages)
            end_time = time.time()

            success = "MODEL_TEST_SUCCESS" in completion.text
            self.log_result(
                model,
                "Basic Completion",
                success,
                f"Response: {completion.text[:50]}...",
                {
                    "response_time": f"{end_time - start_time:.2f}s",
                    "input_tokens": completion.usage.tokens.input_tokens,
                    "output_tokens": completion.usage.tokens.output_tokens,
                    "total_tokens": completion.usage.tokens.total_tokens,
                },
            )

            # Test 3: Mathematical Reasoning
            predict = Predict("problem -> answer")
            register_provider(provider, set_default=True)

            result = await predict.forward(problem="What is 17 * 23?")
            expected_answer = 391
            success = result.success and str(expected_answer) in str(
                result.outputs.get("answer", "")
            )

            self.log_result(
                model,
                "Mathematical Reasoning",
                success,
                f"Problem: 17*23, Answer: {result.outputs.get('answer')}",
                {
                    "correct_answer": str(expected_answer) in str(result.outputs.get("answer", "")),
                    "response_length": len(str(result.outputs.get("answer", ""))),
                },
            )

            # Test 4: Chain-of-Thought Reasoning
            cot = ChainOfThought("question -> reasoning, answer")

            result = await cot.forward(
                question="If a car travels 120 miles in 2 hours, what is its average speed?"
            )

            success = (
                result.success
                and result.outputs.get("reasoning")
                and "60" in str(result.outputs.get("answer", ""))
            )

            self.log_result(
                model,
                "Chain-of-Thought Reasoning",
                success,
                f"Reasoning length: {len(str(result.outputs.get('reasoning', '')))} chars, Answer: {result.outputs.get('answer')}",
                {
                    "has_reasoning": bool(result.outputs.get("reasoning")),
                    "correct_speed": "60" in str(result.outputs.get("answer", "")),
                    "reasoning_length": len(str(result.outputs.get("reasoning", ""))),
                },
            )

            # Test 5: Complex Multi-Step Problem
            result = await cot.forward(
                question="A bakery makes 240 cookies. 1/3 are chocolate chip, 1/4 of the remainder are oatmeal, and the rest are sugar cookies. How many sugar cookies are there?"
            )

            # 240 * (1/3) = 80 chocolate chip
            # Remainder: 240 - 80 = 160
            # Oatmeal: 160 * (1/4) = 40
            # Sugar: 160 - 40 = 120
            expected_sugar = 120

            success = result.success and str(expected_sugar) in str(
                result.outputs.get("answer", "")
            )

            self.log_result(
                model,
                "Complex Multi-Step Problem",
                success,
                f"Expected: {expected_sugar}, Got: {result.outputs.get('answer')}",
                {
                    "correct_answer": str(expected_sugar) in str(result.outputs.get("answer", "")),
                    "has_reasoning": bool(result.outputs.get("reasoning")),
                },
            )

        except Exception as e:
            self.log_result(
                model,
                "Model Testing (FAILED)",
                False,
                f"Critical error: {str(e)[:100]}...",
                {"error_type": type(e).__name__},
            )
            return False

        return True

    async def compare_model_performance(self):
        """Compare performance characteristics across models."""
        print("🏁 PERFORMANCE COMPARISON")
        print("-" * 60)

        test_problems = [
            {"name": "Simple Math", "question": "What is 45 + 37?", "expected": "82"},
            {
                "name": "Word Problem",
                "question": "Sarah has 15 apples. She gives away 7 and buys 3 more. How many apples does she have now?",
                "expected": "11",
            },
            {
                "name": "Logic Problem",
                "question": "All roses are flowers. Some flowers are red. Therefore, what can we conclude about roses?",
                "expected_contains": ["some roses might be red", "roses are flowers"],
            },
        ]

        performance_results = {}

        for model in self.models:
            try:
                provider = create_provider("openai", model=model)
                register_provider(provider, set_default=True)
                predict = Predict("question -> answer")

                model_results = {}

                for problem in test_problems:
                    start_time = time.time()
                    result = await predict.forward(question=problem["question"])
                    end_time = time.time()

                    # Check correctness
                    answer = str(result.outputs.get("answer", "")).lower()
                    if "expected" in problem:
                        correct = problem["expected"].lower() in answer
                    elif "expected_contains" in problem:
                        correct = any(exp.lower() in answer for exp in problem["expected_contains"])
                    else:
                        correct = True  # Default to true if no specific expectation

                    model_results[problem["name"]] = {
                        "response_time": end_time - start_time,
                        "correct": correct,
                        "answer": result.outputs.get("answer", ""),
                        "tokens": result.usage.tokens.total_tokens if result.usage else 0,
                    }

                performance_results[model] = model_results

                # Log summary for this model
                avg_time = sum(r["response_time"] for r in model_results.values()) / len(
                    model_results
                )
                total_correct = sum(1 for r in model_results.values() if r["correct"])
                accuracy = total_correct / len(model_results) * 100

                self.log_result(
                    model,
                    "Performance Summary",
                    True,
                    f"Avg response time: {avg_time:.2f}s, Accuracy: {accuracy:.1f}%",
                    {
                        "average_response_time": f"{avg_time:.2f}s",
                        "accuracy": f"{accuracy:.1f}%",
                        "problems_solved": f"{total_correct}/{len(test_problems)}",
                    },
                )

            except Exception as e:
                self.log_result(
                    model,
                    "Performance Testing (FAILED)",
                    False,
                    f"Error: {str(e)[:100]}...",
                    {"error_type": type(e).__name__},
                )

        return performance_results

    async def run_all_model_tests(self):
        """Run comprehensive tests across all models."""

        print("🚀 LogiLLM MULTI-MODEL VALIDATION")
        print(f"Testing against: {', '.join(self.models)}")
        print("=" * 80)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ No OPENAI_API_KEY found in environment")
            return False

        # Test each model individually
        for model in self.models:
            success = await self.test_model_basic_functionality(model)
            if not success:
                print(f"⚠️  {model} basic tests failed, but continuing with other models...")

        # Performance comparison
        await self.compare_model_performance()

        # Print comprehensive summary
        self.print_summary()

        # Return overall success
        all_results = []
        for model_results in self.results.values():
            all_results.extend(model_results)

        if not all_results:
            return False

        overall_success_rate = sum(1 for r in all_results if r["success"]) / len(all_results)
        return overall_success_rate >= 0.8  # 80% success threshold


async def main():
    """Run multi-model validation."""
    validator = MultiModelValidator()
    success = await validator.run_all_model_tests()

    if success:
        print("\n🎉 MULTI-MODEL VALIDATION SUCCESSFUL!")
        print("LogiLLM works reliably across model generations.")
    else:
        print("\n💥 SOME MULTI-MODEL TESTS FAILED!")
        print("Check model compatibility and fix issues before proceeding.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
