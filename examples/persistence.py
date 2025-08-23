#!/usr/bin/env python3
"""Saving and Loading LogiLLM Modules.

This example demonstrates LogiLLM's built-in persistence system:
1. Save optimized modules with module.save()
2. Load modules with Predict.load()
3. Automatic provider configuration
4. Version compatibility and production workflows

LogiLLM modules now have first-class save/load support built-in!

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import json
import os
from pathlib import Path

from logillm.core.optimizers import AccuracyMetric
from logillm.core.predict import Predict
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider


async def main():
    """Demonstrate module persistence."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== LogiLLM Module Persistence ===")

    try:
        # Step 1: Set up provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Define the save path
        save_path = "saved_modules/email_classifier.json"

        # Check if we have a saved module
        if Path(save_path).exists():
            print(f"\n📁 Loading existing module from {save_path}")

            # Load the saved module
            classifier = Predict.load(save_path)

            # Test the loaded module
            test_emails = [
                "I need help canceling my subscription",
                "Where can I find my billing history?",
                "The app keeps crashing on my phone",
            ]

            print("\n🧪 Testing loaded module:")
            for email in test_emails:
                result = await classifier(email=email)
                intent = result.outputs.get("intent", "unknown")
                print(f"  '{email[:40]}...' → {intent}")

            print("\n✅ Loaded module works perfectly!")
            print("This saved you from re-running optimization!")

        else:
            print("\n🎓 No saved module found. Training a new one...")

            # Step 2: Create and train a module
            classifier = Predict("email -> intent: str", provider=provider)

            # Training data
            training_data = [
                {"inputs": {"email": "Please cancel my account"}, "outputs": {"intent": "cancel"}},
                {"inputs": {"email": "I need my receipt"}, "outputs": {"intent": "billing"}},
                {
                    "inputs": {"email": "Help! The app won't start"},
                    "outputs": {"intent": "support"},
                },
                {
                    "inputs": {"email": "Thanks for the great service!"},
                    "outputs": {"intent": "thanks"},
                },
                {
                    "inputs": {"email": "When will my order arrive?"},
                    "outputs": {"intent": "shipping"},
                },
                {
                    "inputs": {"email": "I want to upgrade my plan"},
                    "outputs": {"intent": "upgrade"},
                },
            ]

            # Optimize with few-shot learning
            metric = AccuracyMetric(key="intent")
            optimizer = BootstrapFewShot(metric=metric, max_examples=3)

            print("• Optimizing module (this takes time and API calls)")
            result = await optimizer.optimize(module=classifier, dataset=training_data)

            optimized_classifier = result.optimized_module

            print("✅ Optimization complete!")
            print(f"Score: {result.best_score:.1%}")

            # Step 3: Save the optimized module
            optimized_classifier.save(save_path)

            # Show what was saved
            if hasattr(optimized_classifier, "demo_manager") and optimized_classifier.demo_manager:
                examples = optimized_classifier.demo_manager.demos
                print(f"💾 Saved {len(examples)} learned examples")

            # Test the optimized module
            test_emails = [
                "I need help canceling my subscription",
                "Where can I find my billing history?",
                "The app keeps crashing on my phone",
            ]

            print("\n🧪 Testing optimized module:")
            for email in test_emails:
                result = await optimized_classifier(email=email)
                intent = result.outputs.get("intent", "unknown")
                print(f"  '{email[:40]}...' → {intent}")

        # Step 4: Show what gets saved
        print("\n📋 What gets saved in the file:")
        if Path(save_path).exists():
            with open(save_path) as f:
                saved_data = json.load(f)

            print(f"• Module type: {saved_data['module_type']}")
            print(f"• LogiLLM version: {saved_data['logillm_version']}")

            if saved_data.get("signature"):
                sig_data = saved_data["signature"]
                input_fields = list(sig_data.get("input_fields", {}).keys())
                output_fields = list(sig_data.get("output_fields", {}).keys())
                print(f"• Signature: {' -> '.join([str(input_fields), str(output_fields)])}")

            print(f"• Configuration: {len(saved_data.get('config', {}))} settings")

            # Show demo manager if present
            if "demo_manager" in saved_data:
                demo_data = saved_data["demo_manager"]
                if demo_data and "demos" in demo_data:
                    print(f"• Few-shot examples: {len(demo_data['demos'])} examples")
                    print(f"• Selection strategy: {demo_data.get('selection_strategy', 'best')}")

            # Show provider info if present
            if "provider_config" in saved_data:
                provider_config = saved_data["provider_config"]
                print(f"• Provider: {provider_config.get('name')} ({provider_config.get('model')})")

        # Step 5: Production usage pattern
        print("\n🏭 Production Usage Pattern:")
        print("1. During development/training:")
        print("   • Create and optimize your module")
        print("   • Save it with optimized_module.save('model.json')")
        print("2. In production:")
        print("   • Load with classifier = Predict.load('model.json')")
        print("   • Use immediately - no re-optimization needed!")

        print("\n✅ Your optimized module is ready for production!")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
