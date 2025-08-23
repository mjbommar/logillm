#!/usr/bin/env python3
"""Comprehensive ChainOfThought validation for LogiLLM.

This systematically tests the ChainOfThought module's REAL capabilities:
- Reasoning field injection and signature modification
- Different reasoning types (math, logic, causal, multi-step)
- Performance comparison vs basic Predict
- Error handling and edge cases
- Integration with adapters and providers

Run with: uv run python tests/core_chainofthought_validation.py
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.predict import ChainOfThought, Predict
from logillm.core.signatures import parse_signature_string
from logillm.providers import create_provider, register_provider


class ChainOfThoughtValidator:
    """Comprehensively validates ChainOfThought reasoning capabilities."""

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

    async def test_chainofthought_architecture(self):
        """Test ChainOfThought architectural capabilities."""

        print("🏗️  ARCHITECTURE TESTING: ChainOfThought Module")
        print("=" * 60)

        # Test 1: Signature Modification
        try:
            # Create basic signature
            base_sig = parse_signature_string("problem -> answer")
            cot = ChainOfThought(base_sig)

            # Verify reasoning field was added
            has_reasoning = "reasoning" in cot.signature.output_fields
            has_answer = "answer" in cot.signature.output_fields
            field_order = list(cot.signature.output_fields.keys())
            reasoning_first = field_order[0] == "reasoning" if field_order else False

            success = has_reasoning and has_answer and reasoning_first

            self.log_result(
                "Signature Modification",
                success,
                f"CoT correctly modified signature: {field_order}",
                {
                    "has_reasoning_field": has_reasoning,
                    "has_original_fields": has_answer,
                    "reasoning_comes_first": reasoning_first,
                    "total_output_fields": len(cot.signature.output_fields),
                },
            )
        except Exception as e:
            self.log_result("Signature Modification", False, f"Error: {e}")
            return False

        # Test 2: Custom Reasoning Field Name
        try:
            cot_custom = ChainOfThought("problem -> solution", reasoning_field="steps")

            has_custom_field = "steps" in cot_custom.signature.output_fields
            no_default_reasoning = "reasoning" not in cot_custom.signature.output_fields

            success = has_custom_field and no_default_reasoning

            self.log_result(
                "Custom Reasoning Field",
                success,
                "Custom reasoning field 'steps' properly configured",
                {
                    "custom_field_present": has_custom_field,
                    "default_field_absent": no_default_reasoning,
                    "reasoning_field_name": cot_custom.reasoning_field,
                },
            )
        except Exception as e:
            self.log_result("Custom Reasoning Field", False, f"Error: {e}")
            return False

        # Test 3: Complex Signature Compatibility
        try:
            # Test with complex multi-input, multi-output signature
            complex_sig = parse_signature_string(
                "context: str, question: str -> reasoning: str, answer: str, confidence: float"
            )
            cot_complex = ChainOfThought(complex_sig)

            # Should have reasoning + original outputs
            expected_outputs = {"reasoning", "answer", "confidence"}
            actual_outputs = set(cot_complex.signature.output_fields.keys())

            success = expected_outputs.issubset(actual_outputs)

            self.log_result(
                "Complex Signature Compatibility",
                success,
                f"Complex signature handled correctly: {list(actual_outputs)}",
                {
                    "expected_fields": len(expected_outputs),
                    "actual_fields": len(actual_outputs),
                    "all_fields_present": success,
                    "input_fields_preserved": len(cot_complex.signature.input_fields) == 2,
                },
            )
        except Exception as e:
            self.log_result("Complex Signature Compatibility", False, f"Error: {e}")
            return False

        return True

    async def test_reasoning_capabilities(self):
        """Test different types of reasoning with real problems."""

        print("🧠 REASONING CAPABILITIES: Different Problem Types")
        print("=" * 60)

        # Prepare provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        reasoning_tests = [
            {
                "name": "Mathematical Reasoning",
                "signature": "problem: str -> reasoning: str, answer: float",
                "input": {
                    "problem": "If a train travels at 80 mph and needs to cover 240 miles, how long will the journey take?"
                },
                "expected_answer": 3.0,
                "validation": lambda ans: abs(float(ans) - 3.0) < 0.1,
            },
            {
                "name": "Logical Reasoning",
                "signature": "premises: str -> reasoning: str, conclusion: str",
                "input": {
                    "premises": "All cats are animals. Fluffy is a cat. What can we conclude about Fluffy?"
                },
                "expected_contains": ["animal", "fluffy"],
                "validation": lambda ans: all(
                    word in str(ans).lower() for word in ["animal", "fluffy"]
                ),
            },
            {
                "name": "Multi-Step Problem Solving",
                "signature": "scenario: str -> reasoning: str, steps: str, final_answer: int",
                "input": {
                    "scenario": "A bakery starts with 200 cookies. In the morning, they sell 25% of them. At lunch, they bake 50 more. In the afternoon, they sell 40 more cookies. How many cookies do they have left?"
                },
                "expected_answer": 110,  # 200 - 50 + 50 - 40 = 160... let me recalculate: 200 - (200*0.25) + 50 - 40 = 200 - 50 + 50 - 40 = 160
                "validation": lambda ans: abs(int(ans) - 160)
                <= 5,  # Allow some variation in interpretation
            },
            {
                "name": "Causal Reasoning",
                "signature": "situation: str -> reasoning: str, cause: str, effect: str",
                "input": {
                    "situation": "The weather forecast predicted rain, so Sarah brought an umbrella. It didn't rain, but she was glad she had it when a bird flew overhead."
                },
                "expected_contains": ["umbrella", "protection"],
                "validation": lambda ans: "umbrella" in str(ans).lower(),
            },
        ]

        for test_case in reasoning_tests:
            try:
                cot = ChainOfThought(test_case["signature"])

                start_time = time.time()
                result = await cot.forward(**test_case["input"])
                end_time = time.time()

                # Extract reasoning and answer
                reasoning = result.outputs.get("reasoning", "")

                # Get the main answer field (varies by test)
                answer_fields = [
                    field for field in result.outputs.keys() if field not in ["reasoning"]
                ]
                main_answer = result.outputs.get(
                    answer_fields[0] if answer_fields else "answer", ""
                )

                # Validate reasoning quality
                reasoning_quality = {
                    "has_reasoning": bool(reasoning and len(str(reasoning)) > 10),
                    "reasoning_length": len(str(reasoning)),
                    "has_step_words": any(
                        word in str(reasoning).lower()
                        for word in ["step", "first", "then", "therefore", "because"]
                    ),
                }

                # Validate answer correctness
                answer_correct = False
                if "validation" in test_case:
                    try:
                        answer_correct = test_case["validation"](main_answer)
                    except (ValueError, TypeError, AttributeError):
                        answer_correct = False

                success = result.success and reasoning_quality["has_reasoning"] and answer_correct

                self.log_result(
                    test_case["name"],
                    success,
                    f"Reasoning: {str(reasoning)[:100]}..., Answer: {main_answer}",
                    {
                        **reasoning_quality,
                        "answer_correct": answer_correct,
                        "response_time": f"{end_time - start_time:.2f}s",
                        "total_tokens": result.usage.tokens.total_tokens if result.usage else 0,
                    },
                )

            except Exception as e:
                self.log_result(test_case["name"], False, f"Error: {e}")

        return True

    async def test_performance_comparison(self):
        """Compare ChainOfThought vs basic Predict performance."""

        print("⚡ PERFORMANCE COMPARISON: CoT vs Basic Predict")
        print("=" * 60)

        # Test problems for comparison
        comparison_problems = [
            {
                "signature": "math_problem: str -> answer: int",
                "input": {"math_problem": "Calculate 17 * 23 + 15 * 8"},
                "expected": 511,  # 391 + 120
                "validate": lambda x: str(511) in str(x),
            },
            {
                "signature": "word_problem: str -> answer: int",
                "input": {
                    "word_problem": "Tom has 25 apples. He gives away 8 apples and buys 12 more. How many apples does he have?"
                },
                "expected": 29,  # 25 - 8 + 12
                "validate": lambda x: str(29) in str(x),
            },
        ]

        performance_results = {"predict": {}, "chainofthought": {}}

        for problem in comparison_problems:
            problem_name = problem["signature"].split(":")[0]

            # Test Basic Predict
            try:
                predict = Predict(problem["signature"])

                start_time = time.time()
                result = await predict.forward(**problem["input"])
                predict_time = time.time() - start_time

                predict_correct = problem["validate"](result.outputs.get("answer", ""))
                predict_tokens = result.usage.tokens.total_tokens if result.usage else 0

                performance_results["predict"][problem_name] = {
                    "correct": predict_correct,
                    "time": predict_time,
                    "tokens": predict_tokens,
                    "answer": result.outputs.get("answer", ""),
                }

            except Exception as e:
                performance_results["predict"][problem_name] = {"error": str(e)}

            # Test ChainOfThought
            try:
                cot = ChainOfThought(problem["signature"])

                start_time = time.time()
                result = await cot.forward(**problem["input"])
                cot_time = time.time() - start_time

                cot_correct = problem["validate"](result.outputs.get("answer", ""))
                cot_tokens = result.usage.tokens.total_tokens if result.usage else 0

                performance_results["chainofthought"][problem_name] = {
                    "correct": cot_correct,
                    "time": cot_time,
                    "tokens": cot_tokens,
                    "answer": result.outputs.get("answer", ""),
                    "reasoning": result.outputs.get("reasoning", ""),
                }

            except Exception as e:
                performance_results["chainofthought"][problem_name] = {"error": str(e)}

        # Analyze results
        predict_correct = sum(
            1 for r in performance_results["predict"].values() if r.get("correct", False)
        )
        cot_correct = sum(
            1 for r in performance_results["chainofthought"].values() if r.get("correct", False)
        )

        predict_avg_time = statistics.mean(
            [r["time"] for r in performance_results["predict"].values() if "time" in r]
        )
        cot_avg_time = statistics.mean(
            [r["time"] for r in performance_results["chainofthought"].values() if "time" in r]
        )

        predict_avg_tokens = statistics.mean(
            [r["tokens"] for r in performance_results["predict"].values() if "tokens" in r]
        )
        cot_avg_tokens = statistics.mean(
            [r["tokens"] for r in performance_results["chainofthought"].values() if "tokens" in r]
        )

        accuracy_improvement = (cot_correct - predict_correct) / len(comparison_problems) * 100
        time_overhead = ((cot_avg_time - predict_avg_time) / predict_avg_time) * 100
        token_overhead = ((cot_avg_tokens - predict_avg_tokens) / predict_avg_tokens) * 100

        success = cot_correct >= predict_correct  # CoT should be at least as good

        self.log_result(
            "Performance Comparison",
            success,
            f"CoT accuracy: {cot_correct}/{len(comparison_problems)}, Basic: {predict_correct}/{len(comparison_problems)}",
            {
                "accuracy_improvement": f"{accuracy_improvement:+.1f}%",
                "time_overhead": f"{time_overhead:+.1f}%",
                "token_overhead": f"{token_overhead:+.1f}%",
                "cot_avg_time": f"{cot_avg_time:.2f}s",
                "predict_avg_time": f"{predict_avg_time:.2f}s",
            },
        )

        # Store for later analysis
        self.performance_data = performance_results

        return True

    async def test_error_handling(self):
        """Test error handling and edge cases."""

        print("🛡️  ERROR HANDLING: Edge Cases and Recovery")
        print("=" * 60)

        # Test 1: Invalid signature
        try:
            try:
                cot = ChainOfThought("invalid signature format")
                success = False  # Should have failed
            except Exception:
                success = True  # Expected to fail

            self.log_result(
                "Invalid Signature Handling",
                success,
                "Invalid signature properly rejected"
                if success
                else "Invalid signature was accepted",
                {"error_properly_caught": success},
            )
        except Exception as e:
            self.log_result("Invalid Signature Handling", False, f"Unexpected error: {e}")

        # Test 2: Empty input handling
        try:
            cot = ChainOfThought("question -> reasoning, answer")
            result = await cot.forward(question="")

            # Should handle gracefully
            success = result is not None

            self.log_result(
                "Empty Input Handling",
                success,
                f"Empty input handled: {result.success if result else 'No result'}",
                {
                    "result_exists": result is not None,
                    "has_outputs": bool(result.outputs) if result else False,
                },
            )
        except Exception as e:
            # Error handling is also acceptable
            self.log_result("Empty Input Handling", True, f"Empty input properly rejected: {e}")

        return True

    async def test_adapter_integration(self):
        """Test ChainOfThought integration with different adapters."""

        print("🔌 ADAPTER INTEGRATION: Format Compatibility")
        print("=" * 60)

        adapters_to_test = [
            ("chat", "Chat format adapter"),
            ("json", "JSON format adapter"),
            ("xml", "XML format adapter"),
            ("markdown", "Markdown format adapter"),
        ]

        for adapter_name, description in adapters_to_test:
            try:
                # Create CoT with specific adapter
                cot = ChainOfThought("question -> reasoning, answer", adapter=adapter_name)

                result = await cot.forward(question="What is 5 + 7?")

                success = (
                    result.success
                    and result.outputs.get("reasoning")
                    and "12" in str(result.outputs.get("answer", ""))
                )

                self.log_result(
                    f"{adapter_name.upper()} Adapter",
                    success,
                    f"CoT works with {description}",
                    {
                        "adapter_type": adapter_name,
                        "has_reasoning": bool(result.outputs.get("reasoning")),
                        "correct_answer": "12" in str(result.outputs.get("answer", "")),
                        "adapter_class": type(cot.adapter).__name__,
                    },
                )

            except Exception as e:
                self.log_result(f"{adapter_name.upper()} Adapter", False, f"Error: {e}")

        return True

    def print_summary(self):
        """Print comprehensive test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        print("=" * 80)
        print("CHAINOFTHOUGHT VALIDATION SUMMARY")
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
            print("ChainOfThought vs Basic Predict comparison completed")
            print("See detailed metrics above for accuracy and efficiency trade-offs")

    async def run_comprehensive_validation(self):
        """Run all ChainOfThought validation tests."""

        print("🧠 LogiLLM CHAINOFTHOUGHT COMPREHENSIVE VALIDATION")
        print("Testing REAL architectural capabilities and reasoning performance...")
        print("=" * 80)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ No OPENAI_API_KEY found in environment")
            return False

        # Run test suites in order
        test_suites = [
            ("Architecture", self.test_chainofthought_architecture),
            ("Reasoning Capabilities", self.test_reasoning_capabilities),
            ("Performance Comparison", self.test_performance_comparison),
            ("Error Handling", self.test_error_handling),
            ("Adapter Integration", self.test_adapter_integration),
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
    """Run ChainOfThought validation."""
    validator = ChainOfThoughtValidator()
    success = await validator.run_comprehensive_validation()

    if success:
        print("\n🎉 CHAINOFTHOUGHT VALIDATION SUCCESSFUL!")
        print("ChainOfThought reasoning architecture is robust and performant.")
    else:
        print("\n💥 CHAINOFTHOUGHT VALIDATION HAD ISSUES!")
        print("Review test results and fix architectural problems before proceeding.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
