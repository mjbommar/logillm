#!/usr/bin/env python3
"""
Example demonstrating complete request/response logging in debug mode.

This example shows how to enable debug mode to capture the complete
request and response data that goes through the LLM provider API.
"""

import asyncio
from logillm.core.predict import Predict
from logillm.providers.mock import MockProvider


async def main():
    """Demonstrate debug mode with complete request/response logging."""

    # Create a mock provider for demonstration
    provider = MockProvider()

    # Create a Predict module with debug mode enabled
    predict = Predict("question -> answer", provider=provider, debug=True)

    print("=== Debug Mode Example: Complete Request/Response Logging ===\n")

    # Make a prediction
    result = await predict(question="What is the capital of France?")

    print("1. Basic Result:")
    print(f"   Success: {result.success}")
    print(f"   Answer: {result.outputs.get('answer', 'N/A')}")
    print()

    print("2. Debug Information Available:")
    print(f"   Has prompt: {result.prompt is not None}")
    print(f"   Has request: {result.request is not None}")
    print(f"   Has response: {result.response is not None}")
    print()

    if result.prompt:
        print("3. Prompt Information:")
        print(f"   Messages count: {len(result.prompt.get('messages', []))}")
        print(f"   Adapter: {result.prompt.get('adapter', 'N/A')}")
        print(f"   Demos count: {result.prompt.get('demos_count', 0)}")
        print(f"   Provider: {result.prompt.get('provider', 'N/A')}")
        print(f"   Model: {result.prompt.get('model', 'N/A')}")
        print()

    if result.request:
        print("4. Complete Request Data:")
        print(f"   Messages: {len(result.request.get('messages', []))} messages")
        print(f"   Provider: {result.request.get('provider', 'N/A')}")
        print(f"   Model: {result.request.get('model', 'N/A')}")
        print(f"   Adapter: {result.request.get('adapter', 'N/A')}")
        print(f"   Demos count: {result.request.get('demos_count', 0)}")
        print(f"   Provider config: {result.request.get('provider_config', {})}")
        print(f"   Timestamp: {result.request.get('timestamp', 'N/A')}")
        print()

    if result.response:
        print("5. Complete Response Data:")
        print(f"   Text: {result.response.get('text', 'N/A')[:100]}...")
        print(f"   Finish reason: {result.response.get('finish_reason', 'N/A')}")
        print(f"   Model: {result.response.get('model', 'N/A')}")
        print(f"   Provider: {result.response.get('provider', 'N/A')}")
        print(f"   Timestamp: {result.response.get('timestamp', 'N/A')}")

        usage = result.response.get("usage", {})
        if usage:
            print("   Usage:")
            print(f"     Input tokens: {usage.get('input_tokens', 0)}")
            print(f"     Output tokens: {usage.get('output_tokens', 0)}")
            print(f"     Cached tokens: {usage.get('cached_tokens', 0)}")
            print(f"     Reasoning tokens: {usage.get('reasoning_tokens', 0)}")
            print(f"     Total tokens: {usage.get('total_tokens', 0)}")
            print(f"     Cost: {result.response.get('cost', 'N/A')}")
            print(f"     Latency: {result.response.get('latency', 'N/A')}")

        metadata = result.response.get("metadata", {})
        if metadata:
            print(f"   Metadata: {metadata}")
        print()

    print("6. Usage Information (also available in result.usage):")
    if result.usage:
        print(f"   Total tokens: {result.usage.tokens.total_tokens if result.usage.tokens else 0}")
        print(f"   Cost: {result.usage.cost}")
        print(f"   Latency: {result.usage.latency}")
        print(f"   Provider: {result.usage.provider}")
        print(f"   Model: {result.usage.model}")
    print()

    print("=== Comparison: Debug Disabled ===")

    # Create another module with debug disabled
    predict_no_debug = Predict("question -> answer", provider=provider, debug=False)
    result_no_debug = await predict_no_debug(question="What is 2+2?")

    print("With debug disabled:")
    print(f"   Has prompt: {result_no_debug.prompt is not None}")
    print(f"   Has request: {result_no_debug.request is not None}")
    print(f"   Has response: {result_no_debug.response is not None}")
    print()

    print("=== Dynamic Debug Toggle ===")

    # Show how to toggle debug mode dynamically
    predict_dynamic = Predict("question -> answer", provider=provider, debug=False)
    result_before = await predict_dynamic(question="Before toggle")
    print(
        f"Before enabling debug: request={result_before.request is not None}, response={result_before.response is not None}"
    )

    predict_dynamic.enable_debug_mode()
    result_after = await predict_dynamic(question="After toggle")
    print(
        f"After enabling debug: request={result_after.request is not None}, response={result_after.response is not None}"
    )

    predict_dynamic.disable_debug_mode()
    result_final = await predict_dynamic(question="After disable")
    print(
        f"After disabling debug: request={result_final.request is not None}, response={result_final.response is not None}"
    )


if __name__ == "__main__":
    asyncio.run(main())
