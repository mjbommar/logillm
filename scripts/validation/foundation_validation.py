#!/usr/bin/env python3
"""Foundation validation tests for LogiLLM.

These tests validate the absolute foundation of LogiLLM with real API calls.
They build confidence step by step in our core abstractions.

Run with: uv run python tests/foundation_validation.py
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.predict import ChainOfThought, Predict
from logillm.providers import create_provider, get_provider, register_provider


class FoundationValidator:
    """Validates LogiLLM foundations with real API calls."""

    def __init__(self):
        """Initialize validator."""
        self.results = []
        self.provider = None

    def log_result(
        self, test_name: str, success: bool, details: str = "", metrics: Dict[str, Any] = None
    ):
        """Log test result."""
        status = "✅" if success else "❌"
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "metrics": metrics or {},
        }
        self.results.append(result)
        print(f"{status} {test_name}: {details}")
        if metrics:
            for key, value in metrics.items():
                print(f"   📊 {key}: {value}")
        print()

    def print_summary(self):
        """Print test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        print("=" * 60)
        print("FOUNDATION VALIDATION SUMMARY")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed / total * 100:.1f}%")
        print("=" * 60)

        if passed < total:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")

    async def test_foundation_layer_1_provider_basics(self):
        """Test Layer 1: Provider creation, registration, basic functionality."""

        print("🧪 FOUNDATION LAYER 1: Provider Basics")
        print("-" * 50)

        # Test 1.1: Provider Creation
        try:
            provider = create_provider("openai", model="gpt-4.1")  # Current model for testing
            self.provider = provider
            self.log_result(
                "1.1 Provider Creation",
                True,
                f"Created {provider.name} provider with model {provider.model}",
                {"provider_name": provider.name, "model": provider.model},
            )
        except Exception as e:
            self.log_result("1.1 Provider Creation", False, f"Error: {e}")
            return False

        # Test 1.2: Provider Registration
        try:
            register_provider(provider, set_default=True)
            retrieved = get_provider()
            success = retrieved is provider
            self.log_result(
                "1.2 Provider Registration",
                success,
                "Provider registered and retrieved as default",
                {"registered": True, "default_set": True},
            )
        except Exception as e:
            self.log_result("1.2 Provider Registration", False, f"Error: {e}")
            return False

        # Test 1.3: Basic Completion
        try:
            messages = [{"role": "user", "content": "Say exactly: FOUNDATION_TEST_SUCCESS"}]
            start_time = time.time()
            completion = await provider.complete(messages)
            end_time = time.time()

            success = "FOUNDATION_TEST_SUCCESS" in completion.text
            self.log_result(
                "1.3 Basic Completion",
                success,
                f"Response: {completion.text[:100]}...",
                {
                    "response_time": f"{end_time - start_time:.2f}s",
                    "input_tokens": completion.usage.tokens.input_tokens,
                    "output_tokens": completion.usage.tokens.output_tokens,
                    "total_tokens": completion.usage.tokens.total_tokens,
                },
            )
        except Exception as e:
            self.log_result("1.3 Basic Completion", False, f"Error: {e}")
            return False

        # Test 1.4: Provider Capabilities
        try:
            capabilities = {
                "streaming": provider.supports_streaming(),
                "structured_output": provider.supports_structured_output(),
                "function_calling": provider.supports_function_calling(),
                "vision": provider.supports_vision(),
            }

            self.log_result(
                "1.4 Provider Capabilities", True, "Capability detection working", capabilities
            )
        except Exception as e:
            self.log_result("1.4 Provider Capabilities", False, f"Error: {e}")
            return False

        return True

    async def test_foundation_layer_2_signatures_predict(self):
        """Test Layer 2: Signatures and Predict module."""

        print("🧪 FOUNDATION LAYER 2: Signatures & Predict")
        print("-" * 50)

        # Test 2.1: Simple String Signature
        try:
            predict = Predict("question -> answer")

            result = await predict.forward(question="What is 7 * 8?")

            success = result.success and "56" in str(result.outputs.get("answer", ""))
            self.log_result(
                "2.1 Simple String Signature",
                success,
                f"Question: 7*8, Answer: {result.outputs.get('answer')}",
                {
                    "success": result.success,
                    "has_answer": "answer" in result.outputs,
                    "correct_math": "56" in str(result.outputs.get("answer", "")),
                },
            )
        except Exception as e:
            self.log_result("2.1 Simple String Signature", False, f"Error: {e}")
            return False

        # Test 2.2: Typed String Signature (simpler alternative to class-based)
        try:
            predict = Predict("problem: str, topic: str -> reasoning: str, answer: int")

            result = await predict.forward(problem="Calculate 15 * 4", topic="arithmetic")

            success = result.success and "60" in str(result.outputs.get("answer", ""))
            self.log_result(
                "2.2 Typed String Signature",
                success,
                f"Problem: 15*4, Reasoning: {str(result.outputs.get('reasoning', ''))[:50]}..., Answer: {result.outputs.get('answer')}",
                {
                    "success": result.success,
                    "has_reasoning": bool(result.outputs.get("reasoning")),
                    "answer_contains_60": "60" in str(result.outputs.get("answer", "")),
                },
            )
        except Exception as e:
            self.log_result("2.2 Typed String Signature", False, f"Error: {e}")
            return False

        # Test 2.3: Multiple Input/Output Fields
        try:
            predict = Predict("name: str, age: int -> greeting: str, category: str")

            result = await predict.forward(name="Alice", age=25)

            success = (
                result.success
                and "Alice" in str(result.outputs.get("greeting", ""))
                and result.outputs.get("category")
            )
            self.log_result(
                "2.3 Multiple Fields",
                success,
                f"Greeting: {result.outputs.get('greeting')}, Category: {result.outputs.get('category')}",
                {
                    "success": result.success,
                    "has_greeting": bool(result.outputs.get("greeting")),
                    "has_category": bool(result.outputs.get("category")),
                    "name_in_greeting": "Alice" in str(result.outputs.get("greeting", "")),
                },
            )
        except Exception as e:
            self.log_result("2.3 Multiple Fields", False, f"Error: {e}")
            return False

        return True

    async def test_foundation_layer_3_parameters_error_handling(self):
        """Test Layer 3: Parameter handling and error scenarios."""

        print("🧪 FOUNDATION LAYER 3: Parameters & Error Handling")
        print("-" * 50)

        # Test 3.1: Parameter Specifications
        try:
            param_specs = self.provider.get_param_specs()

            expected_params = ["temperature", "max_tokens", "top_p"]
            has_expected = all(param in param_specs for param in expected_params)

            self.log_result(
                "3.1 Parameter Specifications",
                has_expected,
                f"Found {len(param_specs)} parameter specs",
                {
                    "total_specs": len(param_specs),
                    "has_temperature": "temperature" in param_specs,
                    "has_max_tokens": "max_tokens" in param_specs,
                    "has_top_p": "top_p" in param_specs,
                },
            )
        except Exception as e:
            self.log_result("3.1 Parameter Specifications", False, f"Error: {e}")
            return False

        # Test 3.2: Parameter Cleaning
        try:
            raw_params = {
                "temperature": "0.7",  # String that should convert
                "max_tokens": 50,
                "invalid_param": "ignored",
                "top_p": 1.1,  # Out of range, should be clamped or filtered
            }

            cleaned = self.provider.clean_params(raw_params)

            success = (
                isinstance(cleaned.get("temperature"), float) and cleaned.get("max_tokens") == 50
            )
            self.log_result(
                "3.2 Parameter Cleaning",
                success,
                f"Cleaned {len(raw_params)} -> {len(cleaned)} valid params",
                {
                    "input_params": len(raw_params),
                    "output_params": len(cleaned),
                    "temp_converted": isinstance(cleaned.get("temperature"), float),
                    "invalid_removed": "invalid_param" not in cleaned,
                },
            )
        except Exception as e:
            self.log_result("3.2 Parameter Cleaning", False, f"Error: {e}")
            return False

        # Test 3.3: Real API with Parameters
        try:
            predict = Predict("task -> response")

            # Parameters should be passed to the provider, not forward()
            # Let's test this by making a direct provider call with parameters
            provider = get_provider()
            messages = [{"role": "user", "content": "Count from 1 to 3, separated by commas"}]

            completion = await provider.complete(
                messages,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=20,
            )

            success = completion.text and ("1" in completion.text and "2" in completion.text)

            # Now test predict module with just signature inputs
            result = await predict.forward(task="Count from 1 to 3, separated by commas")

            response_text = str(result.outputs.get("response", ""))
            predict_success = result.success and ("1" in response_text and "2" in response_text)

            overall_success = success and predict_success

            self.log_result(
                "3.3 Real API with Parameters",
                overall_success,
                f"Provider response: {completion.text[:50]}..., Predict response: {response_text[:50]}...",
                {
                    "provider_success": success,
                    "predict_success": predict_success,
                    "provider_tokens": completion.usage.tokens.total_tokens,
                    "has_counting": bool("1" in completion.text and "2" in completion.text),
                },
            )
        except Exception as e:
            self.log_result("3.3 Real API with Parameters", False, f"Error: {e}")
            return False

        # Test 3.4: Error Handling (Rate Limits, Invalid Requests)
        try:
            # Test with extremely high max_tokens to potentially trigger validation
            predict = Predict("simple -> response")

            result = await predict.forward(
                simple="Hi",
                max_tokens=1000000,  # Extremely high, should be handled gracefully
            )

            # This should either succeed with clamped parameters or fail gracefully
            success = True  # We're testing that it doesn't crash
            self.log_result(
                "3.4 Parameter Validation",
                success,
                "Extreme parameters handled gracefully",
                {"completed_without_crash": True},
            )
        except Exception as e:
            # Expected behavior - validation error should be handled gracefully
            success = "validation" in str(e).lower() or "parameter" in str(e).lower()
            self.log_result(
                "3.4 Parameter Validation",
                success,
                f"Parameter validation error (expected): {str(e)[:100]}...",
                {"graceful_error_handling": success},
            )

        return True

    async def test_foundation_layer_4_chain_of_thought(self):
        """Test Layer 4: ChainOfThought reasoning."""

        print("🧪 FOUNDATION LAYER 4: Chain-of-Thought Reasoning")
        print("-" * 50)

        # Test 4.1: Basic Chain-of-Thought
        try:
            cot = ChainOfThought("problem -> answer")

            result = await cot.forward(
                problem="If a train travels 60 miles in 1.5 hours, what is its average speed?"
            )

            success = (
                result.success
                and "40" in str(result.outputs.get("answer", ""))
                and result.outputs.get("reasoning")
            )

            self.log_result(
                "4.1 Basic Chain-of-Thought",
                success,
                f"Reasoning: {str(result.outputs.get('reasoning', ''))[:100]}..., Answer: {result.outputs.get('answer')}",
                {
                    "success": result.success,
                    "has_reasoning": bool(result.outputs.get("reasoning")),
                    "correct_speed": "40" in str(result.outputs.get("answer", "")),
                    "reasoning_length": len(str(result.outputs.get("reasoning", ""))),
                },
            )
        except Exception as e:
            self.log_result("4.1 Basic Chain-of-Thought", False, f"Error: {e}")
            return False

        # Test 4.2: Complex Multi-Step Problem
        try:
            # Use typed string signature for complex reasoning
            cot = ChainOfThought("problem: str -> reasoning: str, answer: float")

            result = await cot.forward(
                problem="A store has 150 items. On Monday, 25% are sold. On Tuesday, 30% of the remaining items are sold. How many items are left?"
            )

            # 150 - (25% of 150) = 150 - 37.5 = 112.5
            # 112.5 - (30% of 112.5) = 112.5 - 33.75 = 78.75 ≈ 79
            expected_answer = 78.75
            actual_answer = result.outputs.get("answer")

            # Check if answer is close to expected (allow for different rounding)
            answer_close = False
            if actual_answer:
                try:
                    answer_num = float(actual_answer)
                    answer_close = (
                        abs(answer_num - expected_answer) <= 1
                    )  # Allow rounding differences
                except (ValueError, TypeError):
                    # Try to extract number from text answer
                    import re

                    numbers = re.findall(r"\d+(?:\.\d+)?", str(actual_answer))
                    if numbers:
                        answer_num = float(numbers[-1])  # Use last number found
                        answer_close = abs(answer_num - expected_answer) <= 1

            success = result.success and answer_close

            self.log_result(
                "4.2 Complex Multi-Step Problem",
                success,
                f"Expected ~{expected_answer}, Got: {actual_answer}",
                {
                    "success": result.success,
                    "has_reasoning": bool(result.outputs.get("reasoning")),
                    "answer_close": success,
                    "reasoning_steps": str(result.outputs.get("reasoning", "")).count("Step")
                    + str(result.outputs.get("reasoning", "")).count("step"),
                },
            )
        except Exception as e:
            self.log_result("4.2 Complex Multi-Step Problem", False, f"Error: {e}")
            return False

        return True

    async def run_all_foundation_tests(self):
        """Run all foundation validation tests."""

        print("🚀 LogiLLM FOUNDATION VALIDATION")
        print("Testing real functionality step by step...")
        print("=" * 60)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ No OPENAI_API_KEY found in environment")
            return False

        # Run tests in order
        layer_1_ok = await self.test_foundation_layer_1_provider_basics()
        if not layer_1_ok:
            print("❌ Foundation Layer 1 failed - aborting")
            return False

        layer_2_ok = await self.test_foundation_layer_2_signatures_predict()
        if not layer_2_ok:
            print("❌ Foundation Layer 2 failed - aborting")
            return False

        layer_3_ok = await self.test_foundation_layer_3_parameters_error_handling()
        if not layer_3_ok:
            print("❌ Foundation Layer 3 failed - aborting")
            return False

        layer_4_ok = await self.test_foundation_layer_4_chain_of_thought()
        if not layer_4_ok:
            print("❌ Foundation Layer 4 failed - aborting")
            return False

        # Print final summary
        self.print_summary()

        # Return overall success
        return all(r["success"] for r in self.results)


async def main():
    """Run foundation validation."""
    validator = FoundationValidator()
    success = await validator.run_all_foundation_tests()

    if success:
        print("\n🎉 ALL FOUNDATION TESTS PASSED!")
        print("LogiLLM foundations are solid and ready for advanced testing.")
    else:
        print("\n💥 SOME FOUNDATION TESTS FAILED!")
        print("Fix foundation issues before proceeding to advanced features.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
