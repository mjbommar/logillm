#!/usr/bin/env python3
"""
Example demonstrating debug mode with different LLM providers.

This example shows how debug mode works with various providers like
OpenAI, Anthropic, Google, etc.
"""

import asyncio
import os
from logillm.core.predict import Predict


async def test_provider(provider_name: str, api_key_env: str, model_name: str | None = None):
    """Test debug mode with a specific provider."""

    print(f"\n=== Testing {provider_name} Provider ===")

    # Check for API key
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"❌ {api_key_env} environment variable not found")
        print(f"Skipping {provider_name} test")
        return

    print(f"✅ {api_key_env} found")

    try:
        # Create Predict module with debug mode
        predict = Predict("question -> answer", debug=True)

        # Make a simple prediction
        result = await predict(question=f"What is 2+2? (via {provider_name})")

        print("✅ Prediction successful!")
        print(f"Answer: {result.outputs.get('answer', 'N/A')}")

        # Show debug information
        if result.request:
            print(f"📤 Request captured: {len(result.request.get('messages', []))} messages")
            print(f"🤖 Model: {result.request.get('model', 'N/A')}")
            print(f"🏷️  Provider: {result.request.get('provider', 'N/A')}")

        if result.response:
            usage = result.response.get("usage", {})
            if usage:
                print(f"📊 Tokens: {usage.get('total_tokens', 0)} total")
                if result.response.get("cost"):
                    print(f"💰 Cost: ${result.response.get('cost'):.6f}")

        print("✅ Debug mode working correctly!")

    except Exception as e:
        print(f"❌ Error with {provider_name}: {e}")


async def main():
    """Test debug mode with different providers."""

    print("=== Debug Mode with Different Providers ===\n")

    print("This example demonstrates that debug mode works with all LLM providers.")
    print("Each provider will capture complete request/response data when debug=True.\n")

    # Test different providers
    providers_to_test = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-4"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-3-sonnet-20240229"),
        ("Google", "GOOGLE_API_KEY", "gemini-pro"),
    ]

    for provider_name, api_key_env, model_name in providers_to_test:
        await test_provider(provider_name, api_key_env, model_name)

    print("\n=== Summary ===")
    print("✅ Debug mode captures complete request/response data for ALL providers")
    print("✅ Same API works across OpenAI, Anthropic, Google, and others")
    print("✅ Easy access via result.request and result.response")
    print("✅ No provider-specific code needed!")

    print("\n=== Usage Example ===")
    print("""
# Works with any provider - just set the right API key
predict = Predict("question -> answer", debug=True)
result = await predict(question="What is the capital of France?")

# Access complete debug data
request_data = result.request    # Full request sent to API
response_data = result.response  # Full response from API
prompt_data = result.prompt      # Formatted prompt information
""")


if __name__ == "__main__":
    asyncio.run(main())
