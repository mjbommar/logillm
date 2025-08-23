#!/usr/bin/env python3
"""Error Handling with LogiLLM.

This example demonstrates LogiLLM's error handling capabilities:
1. Automatic retries when LLM calls fail
2. Different retry strategies (immediate, linear, exponential)
3. Configurable retry behavior
4. Building robust production systems

Real-world LLM applications need to handle failures gracefully
and retry failed operations automatically.

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.core.retry import Retry, RetryStrategy
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.providers import create_provider, register_provider


class DataAnalysis(Signature):
    """Analyze data and provide insights."""

    data: str = InputField(desc="Raw data to analyze")
    context: str = InputField(desc="Context about the data")

    summary: str = OutputField(desc="Brief summary of key findings")
    confidence: float = OutputField(desc="Confidence in analysis (0.0 to 1.0)")


async def main():
    """Demonstrate retry capabilities."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== Error Handling with LogiLLM ===")

    try:
        # Step 1: Set up provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Step 2: Demonstrate retry functionality
        print("🔄 Automatic Retry Example")
        print("=" * 30)

        # Create a basic predictor
        analyzer = Predict(signature=DataAnalysis, provider=provider)

        # Wrap it with retry logic
        reliable_analyzer = Retry(
            module=analyzer,
            max_retries=3,
            backoff_multiplier=1.5,  # Exponential backoff multiplier
        )

        sample_data = """
        Sales Data Q4 2024:
        - January: $45,000 (down 12% from December)
        - February: $52,000 (up 15.5% from January)
        - March: $38,000 (down 27% from February)
        - Customer complaints increased by 8%
        - New customer acquisition down 15%
        """

        print("Analyzing sales data with retry protection...")

        try:
            result = await reliable_analyzer(
                data=sample_data, context="Quarterly business review for management team"
            )

            print("✅ Analysis completed successfully!")
            summary = result.outputs.get("summary")
            if summary and str(summary) != "PydanticUndefined":
                print(f"Summary: {summary}")
            else:
                print("Summary: Analysis completed but summary not available")

            confidence = result.outputs.get("confidence")
            if confidence:
                print(f"Confidence: {confidence}")

        except Exception as e:
            print(f"❌ Analysis failed even with retries: {e}")

        print("\n" + "=" * 60 + "\n")

        # Step 3: Demonstrate different retry strategies
        print("⚙️ Different Retry Strategies")
        print("=" * 32)

        # Immediate retry (no delay)
        immediate_retry = Retry(module=analyzer, max_retries=2, strategy=RetryStrategy.IMMEDIATE)

        # Linear backoff
        linear_retry = Retry(
            module=analyzer, max_retries=2, strategy=RetryStrategy.LINEAR, base_delay=0.5
        )

        # Exponential backoff (default)
        exponential_retry = Retry(
            module=analyzer,
            max_retries=2,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=0.5,
            backoff_multiplier=2.0,
        )

        simple_data = "Revenue last month: $50,000 (up 5% from previous month)"

        print("Testing different retry strategies...")
        print("• Immediate: Retries without delay")
        print("• Linear: Delays increase by fixed amount")
        print("• Exponential: Delays double each time")

        for name, retry_module in [
            ("Immediate", immediate_retry),
            ("Linear", linear_retry),
            ("Exponential", exponential_retry),
        ]:
            try:
                result = await retry_module(data=simple_data, context="Quick analysis")
                summary = result.outputs.get("summary", "No summary available")
                if str(summary) != "PydanticUndefined":
                    print(f"✅ {name} strategy: {summary[:50]}...")
                else:
                    print(f"✅ {name} strategy: Completed successfully")
            except Exception as e:
                print(f"❌ {name} strategy failed: {e}")

        # Step 4: Show configuration options
        print("\n" + "=" * 60 + "\n")
        print("🔧 Configuration Options")
        print("=" * 25)

        print("Retry Configuration:")
        print("• max_retries: Number of retry attempts (default: 3)")
        print("• strategy: IMMEDIATE, LINEAR, or EXPONENTIAL")
        print("• base_delay: Starting delay in seconds (default: 1.0)")
        print("• max_delay: Maximum delay in seconds (default: 60.0)")
        print("• backoff_multiplier: How much delay increases (default: 2.0)")

        print("\n✅ Production-Ready Error Handling:")
        print("• Automatic retries prevent temporary failures")
        print("• Different strategies for different needs")
        print("• Configurable delays and limits")
        print("• Built-in error feedback for better results")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
