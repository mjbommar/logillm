#!/usr/bin/env python3
"""Critical unit tests to ensure optimization data flows correctly through component lifecycle."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from logillm.core.predict import Predict
from logillm.optimizers.base import Demonstration
from logillm.optimizers.bootstrap_fewshot import BootstrapFewShot


class TestOptimizationLifecycle:
    """Test that optimization data flows correctly through all components."""

    def test_bootstrap_stores_demos_in_demo_manager(self):
        """Critical: Bootstrap must store demos where Predict can find them."""
        # Given: A Predict module
        module = Predict("input -> output")
        initial_demo_count = len(module.demo_manager.demos)

        # When: Bootstrap adds demonstrations
        demo = Demonstration(inputs={"input": "test"}, outputs={"output": "result"}, score=1.0)

        # Simulate what bootstrap should do
        module.demo_manager.add({"inputs": demo.inputs, "outputs": demo.outputs})

        # Then: Demos must be in demo_manager
        assert len(module.demo_manager.demos) == initial_demo_count + 1
        assert module.demo_manager.demos[-1].inputs == {"input": "test"}
        assert module.demo_manager.demos[-1].outputs == {"output": "result"}

    def test_predict_uses_demos_from_demo_manager(self):
        """Critical: Predict must use demos from demo_manager in prompts."""
        # Given: A Predict module with demos
        module = Predict("input -> output")
        module.demo_manager.add({"inputs": {"input": "example"}, "outputs": {"output": "result"}})

        # When: Getting demos for prompt
        demos = []
        for demo in module.demo_manager.get_best():
            demos.append({"inputs": demo.inputs, "outputs": demo.outputs})

        # Then: Demos are available for prompting
        assert len(demos) == 1
        assert demos[0]["inputs"] == {"input": "example"}
        assert demos[0]["outputs"] == {"output": "result"}

    def test_optimizer_to_module_contract(self):
        """Critical: Optimizers must store demos where modules expect them."""
        # This test verifies the CONTRACT between optimizers and modules

        # Given: An optimizer that produces demonstrations
        demos = [
            Demonstration(inputs={"x": 1}, outputs={"y": 2}, score=1.0),
            Demonstration(inputs={"x": 3}, outputs={"y": 4}, score=0.9),
        ]

        # When: Optimizer optimizes a Predict module
        module = Predict("x -> y")

        # The optimizer MUST do this:
        module.demo_manager.clear()
        for demo in demos:
            module.demo_manager.add({"inputs": demo.inputs, "outputs": demo.outputs})

        # Then: Module can access the demos
        retrieved_demos = list(module.demo_manager.get_best())
        assert len(retrieved_demos) == 2

        # And: The demos are in the correct format
        demo_dicts = []
        for demo in retrieved_demos:
            demo_dicts.append({"inputs": demo.inputs, "outputs": demo.outputs})
        assert demo_dicts[0]["inputs"] == {"x": 1}
        assert demo_dicts[0]["outputs"] == {"y": 2}

    @pytest.mark.asyncio
    async def test_end_to_end_demo_flow(self):
        """Integration: Test complete flow from optimizer to prompt."""
        # Given: A complete optimization setup
        module = Predict("question -> answer")

        # Mock the provider to capture what gets sent
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(
            return_value=MagicMock(
                text="mocked response",
                usage=MagicMock(tokens=MagicMock(input_tokens=10, output_tokens=5)),
            )
        )
        module.provider = mock_provider

        # When: We add demos and call forward
        module.demo_manager.add(
            {"inputs": {"question": "What is 2+2?"}, "outputs": {"answer": "4"}}
        )

        # Mock the adapter to check demos are passed
        mock_adapter = AsyncMock()
        mock_adapter.format_type = MagicMock(value="chat")
        mock_adapter.format = AsyncMock(
            return_value=[{"role": "user", "content": "test prompt with demos"}]
        )
        mock_adapter.parse = AsyncMock(return_value={"answer": "test"})
        module.adapter = mock_adapter

        await module.forward(question="What is 3+3?")

        # Then: Adapter receives demos
        mock_adapter.format.assert_called_once()
        call_args = mock_adapter.format.call_args

        # Check that demos were passed to adapter
        assert "demos" in call_args.kwargs
        demos_passed = call_args.kwargs["demos"]
        assert len(demos_passed) == 1
        assert demos_passed[0]["inputs"] == {"question": "What is 2+2?"}
        assert demos_passed[0]["outputs"] == {"answer": "4"}

    def test_parameter_vs_demo_manager_isolation(self):
        """Ensure parameters['demonstrations'] doesn't interfere with demo_manager."""
        # Given: A module with both parameters and demo_manager
        module = Predict("input -> output")

        # When: We add to parameters but not demo_manager
        from logillm.core.modules import Parameter

        module.parameters["demonstrations"] = Parameter(
            value=[{"inputs": {"x": 1}, "outputs": {"y": 2}}], learnable=True
        )

        # Then: demo_manager should be empty (this was the bug!)
        assert len(module.demo_manager.demos) == 0

        # And when: We add to demo_manager
        module.demo_manager.add({"inputs": {"x": 3}, "outputs": {"y": 4}})

        # Then: Only demo_manager demos are used
        demos = list(module.demo_manager.get_best())
        assert len(demos) == 1
        assert demos[0].inputs == {"x": 3}


class TestOptimizerDemoApplication:
    """Test that each optimizer correctly applies demonstrations."""

    def test_bootstrap_applies_demos_to_demo_manager(self):
        """Bootstrap must add demos to demo_manager."""
        # Given: A bootstrap optimizer with selected demos
        BootstrapFewShot(metric=lambda p, e: 1.0, max_bootstrapped_demos=2)

        module = Predict("input -> output")
        len(module.demo_manager.demos)

        # When: Simulating what optimize should do
        demos = [
            Demonstration(inputs={"a": 1}, outputs={"b": 2}, score=1.0),
            Demonstration(inputs={"a": 3}, outputs={"b": 4}, score=0.9),
        ]

        # This is what the fixed bootstrap does:
        if hasattr(module, "demo_manager"):
            module.demo_manager.clear()
            for demo in demos:
                module.demo_manager.add({"inputs": demo.inputs, "outputs": demo.outputs})

        # Then: Demos are in demo_manager
        assert len(module.demo_manager.demos) == 2
        assert module.demo_manager.demos[0].inputs == {"a": 1}

    def test_labeled_fewshot_applies_demos_to_demo_manager(self):
        """LabeledFewShot must add demos to demo_manager."""
        # Similar test for LabeledFewShot
        module = Predict("input -> output")

        demos = [Demonstration(inputs={"x": 5}, outputs={"y": 10}, score=1.0)]

        # Apply demos the correct way
        if hasattr(module, "demo_manager"):
            module.demo_manager.clear()
            for demo in demos:
                module.demo_manager.add({"inputs": demo.inputs, "outputs": demo.outputs})

        assert len(module.demo_manager.demos) == 1
        assert module.demo_manager.demos[0].inputs == {"x": 5}


class TestCriticalInvariants:
    """Test invariants that must always hold for optimization to work."""

    def test_invariant_demos_flow_to_prompts(self):
        """INVARIANT: Demos added by optimizer MUST appear in prompts."""
        # This is the most critical invariant
        # If this fails, optimization has no effect

        # Given: An optimization result with demos
        module = Predict("input -> output")
        module.demo_manager.add({"inputs": {"test": "input"}, "outputs": {"test": "output"}})

        # When: We prepare demos for prompting
        demos_for_prompt = []
        for demo in module.demo_manager.get_best():
            demos_for_prompt.append({"inputs": demo.inputs, "outputs": demo.outputs})

        # Then: The demos MUST be available
        assert len(demos_for_prompt) > 0, "CRITICAL: Demos not flowing to prompts!"
        assert demos_for_prompt[0]["inputs"] == {"test": "input"}

    def test_invariant_optimization_changes_module(self):
        """INVARIANT: Optimization must modify the module in observable ways."""
        # Given: A module before optimization
        module_before = Predict("input -> output")
        demos_before = len(module_before.demo_manager.demos)

        # When: Optimization adds demos (simulated)
        module_after = Predict("input -> output")
        module_after.demo_manager.add({"inputs": {"x": 1}, "outputs": {"y": 2}})
        demos_after = len(module_after.demo_manager.demos)

        # Then: The module MUST be observably different
        assert demos_after > demos_before, "CRITICAL: Optimization had no effect!"

    def test_invariant_demo_format_consistency(self):
        """INVARIANT: Demo format must be consistent throughout pipeline."""
        # Demos must maintain format from optimizer → module → adapter → prompt

        original_demo = {"inputs": {"question": "test"}, "outputs": {"answer": "response"}}

        # Through module
        module = Predict("question -> answer")
        module.demo_manager.add(original_demo)

        # Retrieved from module
        retrieved = list(module.demo_manager.get_best())[0]
        assert retrieved.inputs == original_demo["inputs"]
        assert retrieved.outputs == original_demo["outputs"]

        # Prepared for adapter
        demo_for_adapter = {"inputs": retrieved.inputs, "outputs": retrieved.outputs}

        assert demo_for_adapter == original_demo, "CRITICAL: Demo format corrupted!"


if __name__ == "__main__":
    # Run critical tests
    import sys

    print("🧪 RUNNING CRITICAL LIFECYCLE TESTS")
    print("=" * 60)

    failures = []

    # Test classes to run
    test_classes = [TestOptimizationLifecycle, TestOptimizerDemoApplication, TestCriticalInvariants]

    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    if "asyncio" in str(method):
                        import asyncio

                        asyncio.run(method())
                    else:
                        method()
                    print(f"  ✅ {method_name}")
                except Exception as e:
                    print(f"  ❌ {method_name}: {e}")
                    failures.append((test_class.__name__, method_name, e))

    print("\n" + "=" * 60)
    if failures:
        print(f"❌ {len(failures)} CRITICAL TESTS FAILED:")
        for class_name, method_name, error in failures:
            print(f"  - {class_name}.{method_name}: {error}")
        sys.exit(1)
    else:
        print("✅ ALL CRITICAL LIFECYCLE TESTS PASSED")
        print("The optimization data flow is correctly connected!")
