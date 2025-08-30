#!/usr/bin/env python3
"""
Example demonstrating complete request/response logging with real LLM providers.

This example shows how debug mode captures the complete request and response
data when using real providers like OpenAI, Anthropic, etc.
"""

import asyncio
import os
from logillm.core.predict import Predict


async def main():
    """Demonstrate debug mode with a real LLM provider."""

    print("=== Debug Mode with Real LLM Provider ===\n")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not found!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr use a mock provider for testing:")
        print("from logillm.providers.mock import MockProvider")
        print("predict = Predict('question -> answer', provider=MockProvider(), debug=True)")
        return

    print("✅ OpenAI API key found")
    print("🔧 Creating Predict module with debug mode enabled...\n")

    # Create a Predict module with debug mode enabled
    predict = Predict("question -> answer", debug=True)

    print("🤔 Making prediction with debug logging...\n")

    # Make a prediction
    result = await predict(question="What is the capital of France? Explain briefly.")

    print("=== RESULT ===")
    print(f"Success: {result.success}")
    print(f"Answer: {result.outputs.get('answer', 'N/A')}")
    print()

    if result.request:
        print("=== COMPLETE REQUEST DATA ===")
        print(f"Provider: {result.request.get('provider', 'N/A')}")
        print(f"Model: {result.request.get('model', 'N/A')}")
        print(f"Adapter: {result.request.get('adapter', 'N/A')}")
        print(f"Messages count: {len(result.request.get('messages', []))}")
        print(f"Demos count: {result.request.get('demos_count', 0)}")
        print(f"Timestamp: {result.request.get('timestamp', 'N/A')}")

        # Show provider config (without sensitive data)
        config = result.request.get("provider_config", {})
        safe_config = {k: v for k, v in config.items() if not k.lower().endswith("key")}
        if safe_config:
            print(f"Provider config: {safe_config}")

        print()

    if result.response:
        print("=== COMPLETE RESPONSE DATA ===")
        print(f"Text: {result.response.get('text', 'N/A')[:200]}...")
        print(f"Finish reason: {result.response.get('finish_reason', 'N/A')}")
        print(f"Model: {result.response.get('model', 'N/A')}")
        print(f"Provider: {result.response.get('provider', 'N/A')}")
        print(f"Timestamp: {result.response.get('timestamp', 'N/A')}")

        usage = result.response.get("usage", {})
        if usage:
            print("\n📊 Token Usage:")
            print(f"  Input tokens: {usage.get('input_tokens', 0)}")
            print(f"  Output tokens: {usage.get('output_tokens', 0)}")
            print(f"  Cached tokens: {usage.get('cached_tokens', 0)}")
            print(f"  Reasoning tokens: {usage.get('reasoning_tokens', 0)}")
            print(f"  Total tokens: {usage.get('total_tokens', 0)}")
            print(
                f"  Cost: ${result.response.get('cost', 0):.6f}"
                if result.response.get("cost")
                else "  Cost: N/A"
            )
            print(f"  Latency: {result.response.get('latency', 'N/A')}s")

        metadata = result.response.get("metadata", {})
        if metadata:
            print(f"\n📋 Response metadata: {metadata}")

        print()

    if result.prompt:
        print("=== PROMPT INFORMATION ===")
        print(f"Messages in prompt: {len(result.prompt.get('messages', []))}")
        messages = result.prompt.get("messages", [])
        if messages:
            print("First message preview:")
            first_msg = messages[0]
            content = first_msg.get("content", "")
            print(f"  Role: {first_msg.get('role', 'N/A')}")
            print(
                f"  Content: {content[:100]}..." if len(content) > 100 else f"  Content: {content}"
            )
        print()

    print("=== USAGE SUMMARY ===")
    if result.usage:
        print(f"Total tokens: {result.usage.tokens.total_tokens if result.usage.tokens else 0}")
        print(f"Cost: ${result.usage.cost:.6f}" if result.usage.cost else "Cost: N/A")
        print(f"Latency: {result.usage.latency}s" if result.usage.latency else "Latency: N/A")
        print(f"Provider: {result.usage.provider}")
        print(f"Model: {result.usage.model}")
    print()

    print("=== DEBUG MODE COMPARISON ===")
    print("Creating another module with debug DISABLED...\n")

    # Create module with debug disabled
    predict_no_debug = Predict("question -> answer", debug=False)
    result_no_debug = await predict_no_debug(question="What is 2+2?")

    print("With debug DISABLED:")
    print(f"  Has request data: {result_no_debug.request is not None}")
    print(f"  Has response data: {result_no_debug.response is not None}")
    print(f"  Has prompt data: {result_no_debug.prompt is not None}")
    print(f"  Answer: {result_no_debug.outputs.get('answer', 'N/A')}")
    print()

    print("=== DYNAMIC DEBUG TOGGLE ===")
    print("You can also toggle debug mode on existing modules:\n")

    predict_dynamic = Predict("question -> answer", debug=False)
    result_before = await predict_dynamic(question="Before toggle")
    print(
        f"Before: request={result_before.request is not None}, response={result_before.response is not None}"
    )

    predict_dynamic.enable_debug_mode()
    result_after = await predict_dynamic(question="After toggle")
    print(
        f"After enabling: request={result_after.request is not None}, response={result_after.response is not None}"
    )

    predict_dynamic.disable_debug_mode()
    result_final = await predict_dynamic(question="After disable")
    print(
        f"After disabling: request={result_final.request is not None}, response={result_final.response is not None}"
    )

    print("\n✅ Debug mode demonstration complete!")
    print("💡 Tip: Use debug mode during development to inspect LLM interactions")
    print("💡 Tip: Disable debug mode in production for better performance")


if __name__ == "__main__":
    asyncio.run(main())
