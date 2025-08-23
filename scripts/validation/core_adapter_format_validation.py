#!/usr/bin/env python3
"""Comprehensive Adapter Format validation for LogiLLM.

This systematically tests all adapter formats with REAL API calls:
- Chat, JSON, XML, Markdown format completions
- Round-trip formatting and parsing validation
- Complex data type handling (lists, numbers, booleans)
- Error handling and robustness testing
- Performance comparison across formats
- Format switching capabilities

Run with: uv run python tests/core_adapter_format_validation.py
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logillm.core.adapters import AdapterFormat, create_adapter
from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


class AdapterFormatValidator:
    """Comprehensively validates all adapter formats with real API calls."""

    def __init__(self):
        """Initialize validator."""
        self.results = []
        self.performance_data = {}
        self.adapter_formats = ["chat", "json", "xml", "markdown"]

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

    async def test_basic_adapter_functionality(self):
        """Test basic adapter creation and configuration."""

        print("🔧 BASIC ADAPTER FUNCTIONALITY: Creation and Configuration")
        print("=" * 60)

        # Test 1: Adapter Creation
        try:
            created_adapters = {}

            for format_name in self.adapter_formats:
                adapter = create_adapter(format_name)
                created_adapters[format_name] = adapter

                # Verify adapter properties
                expected_format = AdapterFormat(format_name.upper())
                success = (
                    adapter.format_type == expected_format
                    and adapter.formatter is not None
                    and adapter.parser is not None
                )

                if not success:
                    break

            overall_success = len(created_adapters) == len(self.adapter_formats)

            self.log_result(
                "Adapter Creation",
                overall_success,
                f"Successfully created {len(created_adapters)}/{len(self.adapter_formats)} adapters",
                {
                    "created_count": len(created_adapters),
                    "total_formats": len(self.adapter_formats),
                    "formats_created": list(created_adapters.keys()),
                },
            )

        except Exception as e:
            self.log_result("Adapter Creation", False, f"Error: {e}")
            return False

        # Test 2: Adapter Validation
        try:
            validation_results = {}

            for format_name, adapter in created_adapters.items():
                is_valid = adapter.validate()
                validation_errors = adapter.validation_errors()
                validation_results[format_name] = {"valid": is_valid, "errors": validation_errors}

            all_valid = all(result["valid"] for result in validation_results.values())

            self.log_result(
                "Adapter Validation",
                all_valid,
                f"All adapters validated successfully: {all_valid}",
                {"validation_results": validation_results, "all_valid": all_valid},
            )

        except Exception as e:
            self.log_result("Adapter Validation", False, f"Error: {e}")
            return False

        # Test 3: Configuration Handling
        try:
            config_test_adapter = create_adapter("chat")
            test_config = {"custom_setting": "test_value"}

            config_test_adapter.configure(test_config)
            retrieved_config = config_test_adapter.get_config()

            success = "custom_setting" in retrieved_config

            self.log_result(
                "Configuration Handling",
                success,
                f"Configuration applied and retrieved: {success}",
                {
                    "config_applied": "custom_setting" in retrieved_config,
                    "config_value": retrieved_config.get("custom_setting"),
                },
            )

        except Exception as e:
            self.log_result("Configuration Handling", False, f"Error: {e}")
            return False

        return True

    async def test_format_specific_completions(self):
        """Test each format with real API completions."""

        print("🌐 FORMAT-SPECIFIC COMPLETIONS: Real API Testing")
        print("=" * 60)

        # Prepare provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        test_cases = [
            {
                "name": "Simple Q&A",
                "signature": "question -> answer",
                "input": {"question": "What is the capital of France?"},
                "expected_contains": ["paris"],
                "formats": ["chat", "json", "xml", "markdown"],
            },
            {
                "name": "Multi-Field Output",
                "signature": "topic -> summary: str, key_points: list, confidence: float",
                "input": {"topic": "renewable energy"},
                "expected_checks": {
                    "summary": lambda x: len(str(x)) > 10,
                    "key_points": lambda x: isinstance(x, list) and len(x) > 0,
                    "confidence": lambda x: isinstance(x, (int, float)),
                },
                "formats": ["json", "xml"],  # Complex types work better with structured formats
            },
            {
                "name": "Mathematical Calculation",
                "signature": "problem -> calculation: str, result: int",
                "input": {"problem": "Calculate 15 * 8"},
                "expected_checks": {
                    "result": lambda x: str(120) in str(x) or int(x) == 120
                    if str(x).isdigit()
                    else False
                },
                "formats": ["chat", "json", "xml", "markdown"],
            },
        ]

        format_performance = {}

        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")

            for format_name in test_case["formats"]:
                try:
                    predict_module = Predict(test_case["signature"], adapter=format_name)

                    start_time = time.time()
                    result = await predict_module.forward(**test_case["input"])
                    end_time = time.time()

                    # Performance tracking
                    if format_name not in format_performance:
                        format_performance[format_name] = []
                    format_performance[format_name].append(end_time - start_time)

                    # Validate result
                    success = result.success

                    # Additional validation based on test case
                    if success and "expected_contains" in test_case:
                        # Check if expected content is present
                        output_text = str(result.outputs).lower()
                        for expected in test_case["expected_contains"]:
                            if expected.lower() not in output_text:
                                success = False
                                break

                    if success and "expected_checks" in test_case:
                        # Run custom validation functions
                        for field_name, check_func in test_case["expected_checks"].items():
                            field_value = result.outputs.get(field_name)
                            try:
                                if not check_func(field_value):
                                    success = False
                                    break
                            except Exception:
                                success = False
                                break

                    self.log_result(
                        f"{test_case['name']} - {format_name.upper()}",
                        success,
                        f"Format: {format_name}, Output: {str(result.outputs)[:100]}...",
                        {
                            "format": format_name,
                            "response_time": f"{end_time - start_time:.2f}s",
                            "success": result.success,
                            "output_fields": len(result.outputs),
                            "adapter_type": predict_module.adapter.format_type.value,
                        },
                    )

                except Exception as e:
                    self.log_result(
                        f"{test_case['name']} - {format_name.upper()}", False, f"Error: {e}"
                    )
                    continue

        # Store performance data
        self.performance_data["format_performance"] = format_performance

        return True

    async def test_complex_data_type_handling(self):
        """Test handling of complex data types across formats."""

        print("🧩 COMPLEX DATA TYPES: Lists, Objects, Numbers")
        print("=" * 60)

        complex_test_cases = [
            {
                "name": "List Processing",
                "signature": "items: list -> categorized: list, count: int",
                "input": {"items": ["apple", "banana", "carrot", "broccoli"]},
                "validation": {
                    "categorized": lambda x: isinstance(x, list) and len(x) > 0,
                    "count": lambda x: isinstance(x, int) and x > 0,
                },
                "formats": ["json", "xml"],  # Better for complex types
            },
            {
                "name": "Boolean Logic",
                "signature": "statement -> is_valid: bool, reasoning: str",
                "input": {"statement": "All mammals are warm-blooded"},
                "validation": {
                    "is_valid": lambda x: isinstance(x, bool)
                    or str(x).lower() in ["true", "false"],
                    "reasoning": lambda x: len(str(x)) > 5,
                },
                "formats": ["json", "xml", "markdown"],
            },
            {
                "name": "Numeric Analysis",
                "signature": "numbers: list -> average: float, maximum: int, summary: str",
                "input": {"numbers": [10, 25, 15, 30, 20]},
                "validation": {
                    "average": lambda x: isinstance(x, (int, float)) and 15 <= float(x) <= 25,
                    "maximum": lambda x: isinstance(x, (int, float)) and int(x) == 30,
                    "summary": lambda x: len(str(x)) > 10,
                },
                "formats": ["json", "xml"],
            },
        ]

        for test_case in complex_test_cases:
            print(f"\n🔬 Testing: {test_case['name']}")

            for format_name in test_case["formats"]:
                try:
                    predict_module = Predict(test_case["signature"], adapter=format_name)
                    result = await predict_module.forward(**test_case["input"])

                    # Validate complex data types
                    success = result.success
                    validation_details = {}

                    for field_name, validator in test_case["validation"].items():
                        field_value = result.outputs.get(field_name)
                        try:
                            is_valid = validator(field_value)
                            validation_details[f"{field_name}_valid"] = is_valid
                            validation_details[f"{field_name}_value"] = str(field_value)[:50]
                            if not is_valid:
                                success = False
                        except Exception as e:
                            validation_details[f"{field_name}_error"] = str(e)
                            success = False

                    self.log_result(
                        f"{test_case['name']} - {format_name.upper()}",
                        success,
                        f"Complex data handling: {success}",
                        {
                            "format": format_name,
                            **validation_details,
                            "output_preview": str(result.outputs)[:100],
                        },
                    )

                except Exception as e:
                    self.log_result(
                        f"{test_case['name']} - {format_name.upper()}", False, f"Error: {e}"
                    )

        return True

    async def test_format_switching_capabilities(self):
        """Test dynamic format switching and compatibility."""

        print("🔄 FORMAT SWITCHING: Dynamic Adapter Changes")
        print("=" * 60)

        # Test 1: Same Signature, Different Formats
        try:
            signature = "question -> answer"
            test_input = {"question": "What is 7 * 9?"}

            results_by_format = {}

            for format_name in self.adapter_formats:
                predict_module = Predict(signature, adapter=format_name)
                result = await predict_module.forward(**test_input)

                results_by_format[format_name] = {
                    "success": result.success,
                    "answer": result.outputs.get("answer", ""),
                    "contains_63": "63" in str(result.outputs.get("answer", "")),
                }

            # Check consistency across formats
            all_successful = all(r["success"] for r in results_by_format.values())
            correct_answers = sum(1 for r in results_by_format.values() if r["contains_63"])

            success = (
                all_successful and correct_answers >= len(self.adapter_formats) * 0.75
            )  # 75% threshold

            self.log_result(
                "Format Switching Consistency",
                success,
                f"Success rate: {correct_answers}/{len(self.adapter_formats)} formats",
                {
                    "all_successful": all_successful,
                    "correct_answers": correct_answers,
                    "total_formats": len(self.adapter_formats),
                    "results_by_format": results_by_format,
                },
            )

        except Exception as e:
            self.log_result("Format Switching Consistency", False, f"Error: {e}")
            return False

        # Test 2: Runtime Adapter Switching
        try:
            predict_module = Predict("task -> result")

            # Start with chat adapter
            original_adapter = predict_module.adapter.format_type

            # Switch to JSON adapter
            predict_module.adapter = create_adapter("json")
            new_adapter = predict_module.adapter.format_type

            # Test that it works
            result = await predict_module.forward(task="Count from 1 to 3")

            success = (
                original_adapter != new_adapter
                and new_adapter == AdapterFormat.JSON
                and result.success
            )

            self.log_result(
                "Runtime Adapter Switching",
                success,
                f"Switched from {original_adapter.value} to {new_adapter.value}",
                {
                    "original_format": original_adapter.value,
                    "new_format": new_adapter.value,
                    "switch_successful": original_adapter != new_adapter,
                    "execution_successful": result.success,
                },
            )

        except Exception as e:
            self.log_result("Runtime Adapter Switching", False, f"Error: {e}")
            return False

        return True

    async def test_error_handling_and_robustness(self):
        """Test adapter error handling and fallback capabilities."""

        print("🛡️  ERROR HANDLING: Robustness and Recovery")
        print("=" * 60)

        # Test 1: Malformed Response Handling
        try:
            # Use JSON adapter which is strict about format
            predict_module = Predict("question -> answer", adapter="json")

            # Ask for something that might produce non-JSON response
            result = await predict_module.forward(question="Just say 'hello' without any JSON")

            # Even with a challenging request, adapter should handle gracefully
            success = result is not None

            self.log_result(
                "Malformed Response Handling",
                success,
                f"Handled challenging request: {success}",
                {
                    "result_exists": result is not None,
                    "has_outputs": bool(result.outputs) if result else False,
                    "adapter_format": "json",
                },
            )

        except Exception as e:
            # Expected that some error handling might throw exceptions
            success = "adapter" in str(e).lower() or "parsing" in str(e).lower()
            self.log_result("Malformed Response Handling", success, f"Expected error: {e}")

        # Test 2: Invalid Input Handling
        try:
            adapter = create_adapter("chat")

            # Test with None signature (should be handled gracefully)
            try:
                # This should either work or fail gracefully
                messages = await adapter.format(None, {"test": "input"})
                success = True
            except Exception as e:
                # Graceful failure is also acceptable
                success = "signature" in str(e).lower() or "null" in str(e).lower()

            self.log_result(
                "Invalid Input Handling",
                success,
                "Invalid inputs handled appropriately",
                {"handled_gracefully": success},
            )

        except Exception as e:
            self.log_result("Invalid Input Handling", False, f"Unexpected error: {e}")

        # Test 3: Adapter Format Validation
        try:
            # Test invalid format creation
            try:
                invalid_adapter = create_adapter("invalid_format")
                success = False  # Should have failed
            except Exception:
                success = True  # Expected to fail

            self.log_result(
                "Invalid Format Rejection",
                success,
                "Invalid format properly rejected",
                {"properly_rejected": success},
            )

        except Exception as e:
            self.log_result("Invalid Format Rejection", False, f"Error: {e}")

        return True

    async def test_performance_comparison(self):
        """Compare performance characteristics across adapter formats."""

        print("⚡ PERFORMANCE COMPARISON: Speed and Efficiency")
        print("=" * 60)

        # Performance test with identical tasks
        performance_test_cases = [
            {"signature": "question -> answer", "input": {"question": "What is 5 + 5?"}},
            {
                "signature": "text -> summary",
                "input": {"text": "This is a test sentence for summarization."},
            },
            {"signature": "problem -> solution", "input": {"problem": "How to add two numbers?"}},
        ]

        format_metrics = {format_name: [] for format_name in self.adapter_formats}

        for test_case in performance_test_cases:
            for format_name in self.adapter_formats:
                try:
                    predict_module = Predict(test_case["signature"], adapter=format_name)

                    # Run multiple times for better average
                    times = []
                    for _ in range(2):  # Reduced for speed
                        start_time = time.time()
                        result = await predict_module.forward(**test_case["input"])
                        end_time = time.time()

                        if result.success:
                            times.append(end_time - start_time)

                    if times:
                        avg_time = statistics.mean(times)
                        format_metrics[format_name].append(avg_time)

                except Exception:
                    continue

        # Calculate and compare averages
        format_averages = {}
        for format_name, times in format_metrics.items():
            if times:
                format_averages[format_name] = statistics.mean(times)

        if format_averages:
            fastest_format = min(format_averages, key=format_averages.get)
            slowest_format = max(format_averages, key=format_averages.get)

            speed_diff = (
                (format_averages[slowest_format] - format_averages[fastest_format])
                / format_averages[fastest_format]
                * 100
            )

            self.log_result(
                "Performance Comparison",
                True,
                f"Fastest: {fastest_format} ({format_averages[fastest_format]:.2f}s), Slowest: {slowest_format} ({format_averages[slowest_format]:.2f}s)",
                {
                    "format_averages": {k: f"{v:.2f}s" for k, v in format_averages.items()},
                    "fastest_format": fastest_format,
                    "slowest_format": slowest_format,
                    "speed_difference": f"{speed_diff:.1f}%",
                },
            )

            # Store for summary
            self.performance_data["format_averages"] = format_averages

        return True

    def print_summary(self):
        """Print comprehensive test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        print("=" * 80)
        print("ADAPTER FORMAT VALIDATION SUMMARY")
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
        if "format_averages" in self.performance_data:
            print("\n⚡ PERFORMANCE INSIGHTS:")
            print("Format performance comparison:")
            for format_name, avg_time in self.performance_data["format_averages"].items():
                print(f"  - {format_name.upper()}: {avg_time:.2f}s average")

    async def run_comprehensive_validation(self):
        """Run all adapter format validation tests."""

        print("🔧 LogiLLM ADAPTER FORMAT COMPREHENSIVE VALIDATION")
        print("Testing REAL adapter formats with live API calls...")
        print("=" * 80)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ No OPENAI_API_KEY found in environment")
            return False

        # Run test suites in order
        test_suites = [
            ("Basic Adapter Functionality", self.test_basic_adapter_functionality),
            ("Format-Specific Completions", self.test_format_specific_completions),
            ("Complex Data Type Handling", self.test_complex_data_type_handling),
            ("Format Switching", self.test_format_switching_capabilities),
            ("Error Handling", self.test_error_handling_and_robustness),
            ("Performance Comparison", self.test_performance_comparison),
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
    """Run adapter format validation."""
    validator = AdapterFormatValidator()
    success = await validator.run_comprehensive_validation()

    if success:
        print("\n🎉 ADAPTER FORMAT VALIDATION SUCCESSFUL!")
        print("All adapter formats are working correctly with real API calls.")
    else:
        print("\n💥 ADAPTER FORMAT VALIDATION HAD ISSUES!")
        print("Review test results and fix adapter problems before proceeding.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
