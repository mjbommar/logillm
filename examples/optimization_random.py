#!/usr/bin/env python3
"""RandomPrompt: Random Baseline for Optimization Comparison.

RandomPromptOptimizer provides a random baseline for comparing other
optimization methods. It randomly selects prompts, examples, and
hyperparameters to establish what performance looks like without
intelligent optimization.

Key features:
- Random prompt variations as baseline
- Random hyperparameter sampling
- Random example selection
- Statistical baseline for comparison studies
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.optimizers import RandomPromptOptimizer
from logillm.providers import create_provider, register_provider


class TaskPrioritization(Signature):
    """Prioritize tasks based on importance and urgency."""

    task_description: str = InputField(desc="Description of the task")
    deadline: str = InputField(desc="When the task is due")
    impact: str = InputField(desc="Impact level: low, medium, high")
    priority: str = OutputField(desc="Priority: low, medium, high, urgent")
    reasoning: str = OutputField(desc="Explanation for priority level")
    estimated_effort: str = OutputField(desc="Time estimate: quick, moderate, long")


async def main():
    """Demonstrate random optimization as a baseline comparison."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set: export OPENAI_API_KEY=your_key")
        return

    print("=== RandomPrompt: Baseline for Optimization Studies ===\n")

    # Use consistent model for fair comparison
    provider = create_provider("openai", model="gpt-4.1-nano")
    register_provider(provider, set_default=True)

    prioritizer = Predict(TaskPrioritization)

    # Training data with task prioritization examples
    train_data = [
        {
            "inputs": {
                "task_description": "Fix critical security vulnerability in production system",
                "deadline": "today",
                "impact": "high",
            },
            "outputs": {
                "priority": "urgent",
                "reasoning": "Security vulnerability poses immediate risk to system and users",
                "estimated_effort": "moderate",
            },
        },
        {
            "inputs": {
                "task_description": "Update team documentation for new features",
                "deadline": "next week",
                "impact": "medium",
            },
            "outputs": {
                "priority": "medium",
                "reasoning": "Important for team efficiency but not time-critical",
                "estimated_effort": "long",
            },
        },
        {
            "inputs": {
                "task_description": "Organize team lunch for next month",
                "deadline": "3 weeks",
                "impact": "low",
            },
            "outputs": {
                "priority": "low",
                "reasoning": "Nice to have but not essential for business operations",
                "estimated_effort": "quick",
            },
        },
        {
            "inputs": {
                "task_description": "Prepare quarterly business review presentation",
                "deadline": "tomorrow",
                "impact": "high",
            },
            "outputs": {
                "priority": "urgent",
                "reasoning": "High-impact presentation with immediate deadline",
                "estimated_effort": "long",
            },
        },
        {
            "inputs": {
                "task_description": "Research new development tools for evaluation",
                "deadline": "next month",
                "impact": "medium",
            },
            "outputs": {
                "priority": "medium",
                "reasoning": "Valuable for long-term productivity but not urgent",
                "estimated_effort": "moderate",
            },
        },
    ]

    # Test cases for evaluation
    test_cases = [
        {
            "task_description": "Deploy new feature to production environment",
            "deadline": "end of week",
            "impact": "high",
        },
        {
            "task_description": "Clean up old log files taking up disk space",
            "deadline": "whenever",
            "impact": "low",
        },
        {
            "task_description": "Interview candidate for senior developer position",
            "deadline": "tomorrow",
            "impact": "medium",
        },
        {
            "task_description": "Respond to customer complaint about service outage",
            "deadline": "today",
            "impact": "high",
        },
    ]

    # Baseline test (no optimization)
    print("📊 BASELINE (no optimization):")
    print("-" * 40)
    baseline_accuracy = []

    for test in test_cases:
        result = await prioritizer(**test)
        priority = result.outputs.get("priority", "medium")
        reasoning = result.outputs.get("reasoning", "")[:50]
        effort = result.outputs.get("estimated_effort", "moderate")

        # Simple heuristic scoring based on logical priority
        expected_high = test["impact"] == "high" or test["deadline"] in ["today", "tomorrow"]
        expected_low = test["impact"] == "low" and test["deadline"] in ["whenever", "next month"]

        score = 0
        if expected_high and priority in ["high", "urgent"]:
            score = 1
        elif expected_low and priority == "low":
            score = 1
        elif not expected_high and not expected_low and priority == "medium":
            score = 1
        else:
            score = 0.5  # Partially correct

        baseline_accuracy.append(score)

        print(f"Task: {test['task_description'][:45]}...")
        print(f"  → {priority} ({reasoning}...) [effort: {effort}]")

    baseline_avg = sum(baseline_accuracy) / len(baseline_accuracy)
    print(f"\nBaseline accuracy: {baseline_avg:.2f}")

    # Define metric that rewards correct prioritization
    def priority_metric(pred, ref=None):
        """Reward appropriate priority assignments."""
        priority = pred.get("priority", "medium").lower()
        reasoning = pred.get("reasoning", "")
        effort = pred.get("estimated_effort", "moderate")

        # Base score from reasoning quality
        reasoning_score = min(0.5, len(reasoning.split()) / 20)

        # Consistency bonus (effort should match priority)
        consistency_bonus = 0.0
        if priority == "urgent" and effort in ["moderate", "long"]:
            consistency_bonus += 0.2
        elif priority == "low" and effort in ["quick", "moderate"]:
            consistency_bonus += 0.2
        elif priority == "medium":
            consistency_bonus += 0.1

        return reasoning_score + consistency_bonus

    # Apply random optimization (our baseline)
    print("\n🎲 RANDOM optimization (baseline for comparison)...")

    optimizer = RandomPromptOptimizer(
        metric=priority_metric,
        n_trials=20,  # Try 20 random configurations
        sample_examples=True,  # Randomly sample examples
        sample_hyperparameters=True,  # Randomly sample hyperparameters
        seed=42,  # Reproducible randomness
    )

    # Define search spaces for random sampling
    param_space = {
        "temperature": (0.0, 1.5),  # Random temperature
        "top_p": (0.1, 1.0),  # Random nucleus sampling
        "max_tokens": (50, 300),  # Random response length
    }

    result = await optimizer.optimize(
        module=prioritizer, dataset=train_data, param_space=param_space
    )

    random_optimized = result.optimized_module

    # Test randomly optimized version
    print("\n📊 RANDOM OPTIMIZED:")
    print("-" * 40)
    random_accuracy = []

    for test in test_cases:
        result = await random_optimized(**test)
        priority = result.outputs.get("priority", "medium")
        reasoning = result.outputs.get("reasoning", "")[:50]
        effort = result.outputs.get("estimated_effort", "moderate")

        # Same scoring logic
        expected_high = test["impact"] == "high" or test["deadline"] in ["today", "tomorrow"]
        expected_low = test["impact"] == "low" and test["deadline"] in ["whenever", "next month"]

        score = 0
        if expected_high and priority in ["high", "urgent"]:
            score = 1
        elif expected_low and priority == "low":
            score = 1
        elif not expected_high and not expected_low and priority == "medium":
            score = 1
        else:
            score = 0.5

        random_accuracy.append(score)

        print(f"Task: {test['task_description'][:45]}...")
        print(f"  → {priority} ({reasoning}...) [effort: {effort}]")

    random_avg = sum(random_accuracy) / len(random_accuracy)
    improvement = ((random_avg - baseline_avg) / max(baseline_avg, 0.01)) * 100

    # Results
    print("\n" + "=" * 45)
    print("📈 RANDOM BASELINE RESULTS:")
    print(f"  Baseline accuracy:      {baseline_avg:.2f}")
    print(f"  Random optimized:       {random_avg:.2f}")
    print(f"  Random improvement:     {improvement:+.1f}%")

    # Show random configuration discovered
    if hasattr(random_optimized, "config") and random_optimized.config:
        print("\n🎲 Random configuration:")
        if "temperature" in random_optimized.config:
            print(f"  Temperature:  {random_optimized.config['temperature']:.2f}")
        if "top_p" in random_optimized.config:
            print(f"  Top-p:        {random_optimized.config['top_p']:.2f}")
        if "max_tokens" in random_optimized.config:
            print(f"  Max tokens:   {random_optimized.config['max_tokens']}")

    # Show random optimization stats
    if hasattr(result, "metadata"):
        print("\n📊 Random search statistics:")
        if "trials_completed" in result.metadata:
            print(f"  Trials completed:     {result.metadata['trials_completed']}")
        if "best_score" in result.metadata:
            print(f"  Best random score:    {result.metadata['best_score']:.3f}")
        if "configurations_tried" in result.metadata:
            print(f"  Configurations tried: {result.metadata['configurations_tried']}")

    # Show selected random examples
    if hasattr(random_optimized, "demo_manager") and random_optimized.demo_manager.demos:
        print(f"\n📚 Randomly selected examples ({len(random_optimized.demo_manager.demos)}):")
        for i, demo in enumerate(random_optimized.demo_manager.demos[:2], 1):
            task = demo.inputs.get("task_description", "")[:30]
            priority = demo.outputs.get("priority", "")
            print(f"  {i}. '{task}...' → {priority}")

    print("\n✨ Key insight: Random optimization provides a crucial baseline")
    print("   for measuring whether sophisticated optimization methods are")
    print("   actually better than chance. Always compare to random!")


if __name__ == "__main__":
    asyncio.run(main())
