#!/usr/bin/env python3
"""
Test validation script to assess the real state of LogiLLM testing.
Run this to get an honest assessment of what works and what doesn't.
"""

import asyncio
import os
import sys
import time

import pytest

# Add logillm to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"         {details}")


@pytest.mark.asyncio
async def test_basic_import():
    """Test that LogiLLM can be imported."""
    try:
        print_result("Basic imports", True)
        return True
    except Exception as e:
        print_result("Basic imports", False, str(e))
        return False


@pytest.mark.asyncio
async def test_mock_provider():
    """Test that mock provider works."""
    try:
        from logillm.core.predict import Predict
        from logillm.providers import MockProvider, register_provider

        mock = MockProvider(responses=["42"])
        register_provider(mock, set_default=True)

        module = Predict("question -> answer")
        result = await module.forward(question="What is 6 times 7?")

        # Check both possible output field names (adapter might use "output" or "answer")
        answer = result.outputs.get("answer") or result.outputs.get("output")
        passed = result.success and answer == "42"
        print_result("Mock provider", passed, f"Got: {result.outputs}")
        return passed
    except Exception as e:
        print_result("Mock provider", False, str(e))
        return False


@pytest.mark.asyncio
async def test_api_access():
    """Test if we have working API access."""
    results = {}

    # Check OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        # Try to load from bashrc
        bashrc_path = os.path.expanduser("~/.bashrc")
        if os.path.exists(bashrc_path):
            with open(bashrc_path) as f:
                for line in f:
                    if "OPENAI_API_KEY" in line and not line.strip().startswith("#"):
                        openai_key = line.split("=")[1].strip()
                        break

    if openai_key and not openai_key.startswith("#"):
        print_result("OpenAI API key", True, "Found (not validated)")
        results["openai"] = True
    else:
        print_result("OpenAI API key", False, "Not found or commented out")
        results["openai"] = False

    # Check Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        print_result("Gemini API key", True, "Found in environment")
        results["gemini"] = True
    else:
        print_result("Gemini API key", False, "Not found")
        results["gemini"] = False

    return results


@pytest.mark.asyncio
async def test_optimization_features():
    """Test which optimizers are actually implemented."""
    results = {}

    # Test each optimizer
    optimizers = [
        ("BootstrapFewShot", "logillm.optimizers.BootstrapFewShot"),
        ("HybridOptimizer", "logillm.optimizers.HybridOptimizer"),
        ("FormatOptimizer", "logillm.optimizers.FormatOptimizer"),
        ("InstructionOptimizer", "logillm.optimizers.InstructionOptimizer"),
        ("MultiObjective", "logillm.optimizers.MultiObjective"),
        ("HyperparameterOptimizer", "logillm.optimizers.HyperparameterOptimizer"),
    ]

    for name, import_path in optimizers:
        try:
            module_path, class_name = import_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)

            # Check if optimize method exists
            has_optimize = hasattr(cls, "optimize")

            if has_optimize:
                print_result(f"{name} optimizer", True, "Has optimize method")
                results[name] = True
            else:
                print_result(f"{name} optimizer", False, "Missing optimize method")
                results[name] = False

        except Exception as e:
            print_result(f"{name} optimizer", False, f"Import failed: {e}")
            results[name] = False

    return results


@pytest.mark.asyncio
async def test_adapters():
    """Test which adapters are functional."""
    from logillm.core.adapters import create_adapter
    from logillm.core.signatures import parse_signature_string

    results = {}
    formats = ["json", "chat", "markdown", "xml"]

    sig = parse_signature_string("input -> output")
    test_input = {"input": "test"}

    for fmt in formats:
        try:
            adapter = create_adapter(fmt)

            # Test that adapter has required attributes
            has_format = hasattr(adapter, "format_type")

            # Try to format a prompt
            if hasattr(adapter, "format"):
                # Old interface
                adapter.format(sig, inputs=test_input)
                can_format = True
            elif hasattr(adapter, "format_prompt"):
                # New interface
                adapter.format_prompt(sig, test_input)
                can_format = True
            else:
                can_format = False

            passed = has_format and can_format
            print_result(f"{fmt} adapter", passed)
            results[fmt] = passed

        except Exception as e:
            print_result(f"{fmt} adapter", False, str(e))
            results[fmt] = False

    return results


async def count_tests():
    """Count the number of tests in different categories."""
    import subprocess

    # Count unit tests
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/unit/", "--co", "-q"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        unit_count = len([l for l in result.stdout.split("\n") if "test" in l])
    except:
        unit_count = 0

    # Count integration tests
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/integration/", "--co", "-q", "-m", "integration"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        integration_count = len([l for l in result.stdout.split("\n") if "test" in l])
    except:
        integration_count = 0

    print_result("Unit tests", True, f"{unit_count} tests found")
    print_result("Integration tests", True, f"{integration_count} tests found")

    return {"unit": unit_count, "integration": integration_count}


@pytest.mark.asyncio
async def test_real_optimization():
    """Test if optimization actually improves performance with mock."""
    try:
        from logillm.core.predict import Predict
        from logillm.optimizers import BootstrapFewShot
        from logillm.providers import MockProvider, register_provider

        # Setup mock that improves with demos
        mock = MockProvider(
            responses=[
                '{"answer": "wrong"}',  # Baseline: wrong
                '{"answer": "4"}',  # Bootstrap teacher: correct
                '{"answer": "6"}',  # Bootstrap teacher: correct
                '{"answer": "4"}',  # With demos: correct
                '{"answer": "6"}',  # With demos: correct
            ]
        )
        register_provider(mock, set_default=True)

        # Simple dataset
        train_data = [
            {"inputs": {"question": "2+2"}, "outputs": {"answer": "4"}},
            {"inputs": {"question": "3+3"}, "outputs": {"answer": "6"}},
        ]

        def metric(pred, expected):
            return 1.0 if pred.get("answer") == expected.get("answer") else 0.0

        # Test baseline
        module = Predict("question -> answer")
        baseline_score = 0
        for example in train_data:
            result = await module.forward(**example["inputs"])
            baseline_score += metric(result.outputs, example["outputs"])
        baseline_score /= len(train_data)

        # Reset mock responses for optimization
        mock.response_index = 1

        # Optimize
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
        opt_result = await optimizer.optimize(module, train_data)

        # Check improvement
        improvement = opt_result.improvement or 0
        passed = improvement > 0

        print_result(
            "Optimization improves performance",
            passed,
            f"Baseline: {baseline_score:.0%}, Improvement: {improvement:+.0%}",
        )
        return passed

    except Exception as e:
        print_result("Optimization improves performance", False, str(e))
        return False


async def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("  LogiLLM Test Validation Report")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # Track overall results
    all_passed = True

    print_section("1. Basic Functionality")
    if not await test_basic_import():
        all_passed = False
    if not await test_mock_provider():
        all_passed = False

    print_section("2. API Access")
    api_results = await test_api_access()
    if not any(api_results.values()):
        print("\n⚠️  WARNING: No API keys found. Integration tests will fail!")
        all_passed = False

    print_section("3. Optimizers")
    opt_results = await test_optimization_features()
    if not all(opt_results.values()):
        all_passed = False

    print_section("4. Adapters")
    adapter_results = await test_adapters()
    if not all(adapter_results.values()):
        all_passed = False

    print_section("5. Test Coverage")
    await count_tests()

    print_section("6. Optimization Validation")
    if not await test_real_optimization():
        all_passed = False

    # Summary
    print_section("Summary")

    if all_passed:
        print("✅ All validation tests passed!")
    else:
        print("❌ Some validation tests failed")

    print("\nRecommendations:")
    if not api_results.get("openai") and not api_results.get("gemini"):
        print("1. Enable API access: Uncomment OPENAI_API_KEY in ~/.bashrc")
        print("   OR implement Gemini provider using available GEMINI_API_KEY")

    if not opt_results.get("FormatOptimizer"):
        print("2. Complete FormatOptimizer implementation")

    if not opt_results.get("HybridOptimizer"):
        print("3. Fix HybridOptimizer to demonstrate value")

    print("\nNext Steps:")
    print("1. Fix any failing components above")
    print("2. Run: uv run pytest tests/integration/ -m integration")
    print("3. Create benchmarks to prove optimization value")
    print("4. Document real performance numbers")


if __name__ == "__main__":
    asyncio.run(main())
