#!/usr/bin/env python3
"""Comprehensive JSONL logging example showing all features.

This example demonstrates:
- JSONL logging of optimization process
- Capturing hyperparameters from provider configuration
- Recording instructions from signature docstrings
- Capturing actual prompts sent to the LLM
- Tracking optimization metrics and progression
"""

import asyncio
import json
from pathlib import Path

from logillm.core.jsonl_logger import OptimizationLogger
from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers.bootstrap_fewshot import BootstrapFewShot
from logillm.providers import create_provider, register_provider


class MathSignature(Signature):
    """Solve the math problem step by step, showing your work."""

    problem: str = InputField(desc="A math problem to solve")
    solution: str = OutputField(desc="Step-by-step solution with final answer")


class PromptCapturingLogger(OptimizationLogger):
    """Extended logger that captures actual prompts during optimization."""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.captured_prompts = []

    async def log_optimization(self, optimizer, module, dataset, **kwargs):
        """Log optimization with prompt capture."""

        # Store the original forward method
        original_forward = module.forward
        captured_prompts = self.captured_prompts

        async def capturing_forward(**inputs):
            """Wrapped forward that captures prompts."""
            # Enable debug mode temporarily
            original_debug = getattr(module, "_debug_mode", False)
            module._debug_mode = True

            try:
                # Call original forward
                result = await original_forward(**inputs)

                # Capture the prompt if available
                if result.prompt:
                    prompt_info = {
                        "inputs": inputs,
                        "messages": result.prompt.get("messages", []),
                        "adapter": result.prompt.get("adapter"),
                        "demos_count": result.prompt.get("demos_count", 0),
                        "model": result.prompt.get("model"),
                    }
                    captured_prompts.append(prompt_info)

                return result
            finally:
                # Restore debug mode
                module._debug_mode = original_debug

        # Replace forward method
        module.forward = capturing_forward

        try:
            # Run optimization with capturing
            result = await super().log_optimization(optimizer, module, dataset, **kwargs)

            # Add captured prompts to the log
            if self.captured_prompts:
                self._write_event(
                    {
                        "event_type": "prompts_captured",
                        "total_prompts": len(self.captured_prompts),
                        "sample_prompts": self.captured_prompts[:3],  # First 3 examples
                    }
                )

            return result
        finally:
            # Restore original forward
            module.forward = original_forward


async def main():
    """Run comprehensive JSONL logging demonstration."""

    print("=" * 70)
    print("🚀 COMPREHENSIVE JSONL LOGGING DEMONSTRATION")
    print("=" * 70)

    # Create provider with explicit hyperparameters
    try:
        # Try real OpenAI provider
        provider = create_provider(
            "openai",
            model="gpt-4.1-mini",
            temperature=0.7,
            max_tokens=150,
            top_p=0.95,
            frequency_penalty=0.1,
        )
    except Exception:
        # Fallback to mock for demonstration
        print("⚠️  Using mock provider for demonstration")
        from logillm.providers.mock import MockProvider

        provider = MockProvider(
            responses=[
                "Step 1: 15 + 27 = 42",
                "Step 1: 8 × 7 = 56",
                "Step 1: 100 ÷ 4 = 25",
                "Step 1: 12 × 12 = 144",
            ]
        )

    register_provider(provider, set_default=True)

    # Create module with class-based signature (has instructions)
    math_module = Predict(MathSignature)

    # Training dataset
    dataset = [
        {"inputs": {"problem": "What is 15 + 27?"}, "outputs": {"solution": "15 + 27 = 42"}},
        {"inputs": {"problem": "What is 8 × 7?"}, "outputs": {"solution": "8 × 7 = 56"}},
        {"inputs": {"problem": "What is 100 ÷ 4?"}, "outputs": {"solution": "100 ÷ 4 = 25"}},
    ]

    # Validation dataset
    validation = [
        {"inputs": {"problem": "What is 12 × 12?"}, "outputs": {"solution": "12 × 12 = 144"}}
    ]

    # Evaluation metric
    def solution_accuracy(pred, ref):
        """Check if the solution contains the correct answer."""
        import re

        pred_text = pred.get("solution", "").lower()
        ref_text = ref.get("solution", "").lower()

        # Extract numbers from both
        pred_nums = re.findall(r"\d+", pred_text)
        ref_nums = re.findall(r"\d+", ref_text)

        if ref_nums and pred_nums:
            # Check if final answer matches
            return 1.0 if ref_nums[-1] in pred_nums else 0.0
        return 0.0

    # Create optimizer
    optimizer = BootstrapFewShot(
        metric=solution_accuracy, max_demos=2, max_bootstraps=3, max_rounds=2
    )

    # Setup prompt-capturing logger
    log_path = Path("comprehensive_optimization.jsonl")
    logger = PromptCapturingLogger(filepath=str(log_path))

    print(f"\n📝 Logging to: {log_path}")
    print(
        f"🔧 Hyperparameters: temperature={provider.config.get('temperature')}, "
        f"max_tokens={provider.config.get('max_tokens')}"
    )
    print(f"📚 Training samples: {len(dataset)}, Validation samples: {len(validation)}")

    # Run optimization
    result = await logger.log_optimization(
        optimizer=optimizer, module=math_module, dataset=dataset, validation_set=validation
    )

    print("\n✅ Optimization complete!")
    print(f"   Best score: {result.best_score:.2%}")
    print(f"   Improvement: {result.improvement:.2%}")
    print(f"   Time: {result.optimization_time:.2f}s")
    print(f"   Captured prompts: {len(logger.captured_prompts)}")

    # Analyze the JSONL log
    print("\n" + "=" * 70)
    print("📊 JSONL LOG ANALYSIS")
    print("=" * 70)

    with open(log_path) as f:
        events = [json.loads(line) for line in f]

    # Show event types
    event_counts = {}
    for event in events:
        event_type = event.get("event_type", "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    print("\n📈 Event Statistics:")
    for event_type, count in sorted(event_counts.items()):
        print(f"   • {event_type}: {count}")

    # Show initial configuration
    start_event = events[0]
    if start_event.get("event_type") == "optimization_start":
        initial = start_event.get("initial_module", {})
        print("\n🔍 Initial Configuration:")
        print(f"   • Signature: {initial.get('signature', 'N/A')}")
        print(f"   • Instructions: '{initial.get('instructions', 'N/A')}'")

        hparams = initial.get("hyperparameters", {})
        if hparams:
            print("   • Hyperparameters:")
            for key, value in hparams.items():
                if value is not None:
                    print(f"      - {key}: {value}")

    # Show captured prompts
    prompts_event = next((e for e in events if e.get("event_type") == "prompts_captured"), None)
    if prompts_event:
        print("\n💬 Captured Prompts:")
        print(f"   • Total captured: {prompts_event.get('total_prompts', 0)}")

        sample_prompts = prompts_event.get("sample_prompts", [])
        if sample_prompts:
            first_prompt = sample_prompts[0]
            print("\n   📌 Example Prompt:")
            print(f"      Input: {first_prompt['inputs'].get('problem', 'N/A')}")
            print(f"      Demos included: {first_prompt['demos_count']}")
            print(f"      Model: {first_prompt['model']}")

            messages = first_prompt["messages"]
            if messages:
                print(f"      Messages ({len(messages)} total):")
                for i, msg in enumerate(messages[:2], 1):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"         {i}. [{role}]: {content}")

    # Show final configuration
    final_event = events[-1]
    if final_event.get("event_type") == "optimization_end":
        final = final_event.get("final_module", {})
        print("\n🎯 Final Configuration:")
        print(f"   • Demonstrations added: {final.get('num_demos', 0)}")

        if "demo_example" in final:
            demo = final["demo_example"]
            print("   • Example demo:")
            print(f"      Input: {demo['inputs']['problem']}")
            print(f"      Output: {demo['outputs']['solution']}")

    # Show score progression
    eval_scores = []
    for event in events:
        if event.get("event_type") == "evaluation_end" and "score" in event:
            eval_scores.append(event["score"])

    if eval_scores:
        print("\n📈 Score Progression:")
        for i, score in enumerate(eval_scores, 1):
            bar = "█" * int(score * 20)
            print(f"   Round {i}: {bar:<20} {score:.2%}")

    print(f"\n✨ Full log saved to '{log_path}'")
    print("💡 The log contains complete optimization history with prompts!")


if __name__ == "__main__":
    asyncio.run(main())
