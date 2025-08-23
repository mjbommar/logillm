#!/usr/bin/env python3
"""Few-Shot Learning with LogiLLM.

This example demonstrates LogiLLM's bootstrap few-shot capabilities:
1. Starting with a basic module with no examples
2. Using bootstrap learning to automatically generate helpful examples
3. Comparing performance before and after adding examples
4. Understanding how few-shot examples improve results

Few-shot learning means giving the LLM examples of correct input-output pairs
to help it understand what you want. LogiLLM can automatically generate
these examples for you.

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os

from logillm.core.optimizers import AccuracyMetric
from logillm.core.predict import Predict
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider


async def main():
    """Demonstrate few-shot learning."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== Few-Shot Learning with LogiLLM ===")

    try:
        # Step 1: Set up provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Step 2: Create a module without any examples
        # We'll teach it to classify email intent
        email_classifier = Predict("email -> intent: str", provider=provider)

        print("📧 Email Intent Classification")
        print("=" * 35)

        # Test cases to evaluate performance
        test_emails = [
            "Hi, I'd like to cancel my subscription effective immediately.",
            "Could you please send me a copy of my recent invoice?",
            "I'm having trouble logging into my account. The password reset isn't working.",
            "Thank you so much for the quick response! You solved my problem perfectly.",
            "When will my order ship? I placed it 3 days ago.",
            "I want to upgrade my plan to include more storage space.",
        ]

        expected_intents = ["cancel", "billing", "support", "thanks", "shipping", "upgrade"]

        # Step 3: Test baseline performance (no examples)
        print("\n🧪 Testing WITHOUT few-shot examples:")
        print("-" * 40)

        baseline_correct = 0
        baseline_predictions = []

        for email, expected in zip(test_emails, expected_intents):
            result = await email_classifier(email=email)
            predicted = result.outputs.get("intent", "").lower()
            baseline_predictions.append(predicted)

            is_correct = (
                expected.lower() in predicted.lower() or predicted.lower() in expected.lower()
            )
            if is_correct:
                baseline_correct += 1

            status = "✓" if is_correct else "✗"
            print(f"  {status} '{email[:50]}...' → {predicted} (expected: {expected})")

        baseline_accuracy = baseline_correct / len(test_emails)
        print(f"\n📊 Baseline Accuracy: {baseline_accuracy:.1%}")

        # Step 4: Prepare training data for bootstrap learning
        print("\n🎓 Training few-shot examples...")

        training_data = [
            {"inputs": {"email": "Please cancel my account"}, "outputs": {"intent": "cancel"}},
            {
                "inputs": {"email": "I need my receipt from last month"},
                "outputs": {"intent": "billing"},
            },
            {"inputs": {"email": "Help! I can't log in"}, "outputs": {"intent": "support"}},
            {"inputs": {"email": "Thanks for fixing the bug!"}, "outputs": {"intent": "thanks"}},
            {"inputs": {"email": "Where is my package?"}, "outputs": {"intent": "shipping"}},
            {"inputs": {"email": "I want to upgrade to premium"}, "outputs": {"intent": "upgrade"}},
            {"inputs": {"email": "Close my subscription please"}, "outputs": {"intent": "cancel"}},
            {
                "inputs": {"email": "Send me the invoice for order #123"},
                "outputs": {"intent": "billing"},
            },
            {"inputs": {"email": "The app keeps crashing"}, "outputs": {"intent": "support"}},
            {"inputs": {"email": "Perfect solution, thank you!"}, "outputs": {"intent": "thanks"}},
        ]

        # Step 5: Use bootstrap few-shot to improve the module
        metric = AccuracyMetric(key="intent")

        bootstrap = BootstrapFewShot(
            metric=metric,
            max_examples=4,  # Add up to 4 helpful examples
            teacher_module=None,  # Will create a teacher automatically
        )

        print("• Analyzing training data...")
        print("• Generating helpful examples...")
        print("• Testing example combinations...")

        optimized_result = await bootstrap.optimize(module=email_classifier, dataset=training_data)

        improved_classifier = optimized_result.optimized_module

        # Step 6: Test performance with few-shot examples
        print("\n🧪 Testing WITH few-shot examples:")
        print("-" * 38)

        improved_correct = 0
        improved_predictions = []

        for email, expected in zip(test_emails, expected_intents):
            result = await improved_classifier(email=email)
            predicted = result.outputs.get("intent", "").lower()
            improved_predictions.append(predicted)

            is_correct = (
                expected.lower() in predicted.lower() or predicted.lower() in expected.lower()
            )
            if is_correct:
                improved_correct += 1

            status = "✓" if is_correct else "✗"
            print(f"  {status} '{email[:50]}...' → {predicted} (expected: {expected})")

        improved_accuracy = improved_correct / len(test_emails)
        improvement = improved_accuracy - baseline_accuracy

        print("\n📊 Results Summary:")
        print(f"Without examples: {baseline_accuracy:.1%}")
        print(f"With examples:    {improved_accuracy:.1%}")
        print(f"Improvement:      {improvement:+.1%}")

        # Step 7: Show what examples were added
        if hasattr(improved_classifier, "demo_manager") and improved_classifier.demo_manager:
            examples = improved_classifier.demo_manager.demos
            print(f"\n📚 Examples Added ({len(examples)} total):")
            for i, demo in enumerate(examples, 1):
                input_text = demo.inputs.get("email", "")[:30]
                output_text = demo.outputs.get("intent", "")
                print(f"  {i}. '{input_text}...' → {output_text}")

        # Step 8: Demonstrate with new test cases
        print("\n🔮 Testing on new emails:")
        print("-" * 25)

        new_test_cases = [
            "I'd like to return this item for a full refund",
            "Could you help me reset my password?",
            "This is exactly what I needed, amazing work!",
        ]

        for email in new_test_cases:
            result = await improved_classifier(email=email)
            intent = result.outputs.get("intent", "")
            print(f"'{email}' → {intent}")

        print("\n✅ Few-shot learning improves accuracy by teaching the model!")
        print("• Examples show the model what you want")
        print("• Bootstrap learning finds the most helpful examples automatically")
        print("• Performance typically improves significantly")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
