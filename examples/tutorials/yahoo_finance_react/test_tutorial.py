"""Test script for Yahoo Finance ReAct tutorial."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.tutorials.yahoo_finance_react.demo import demo_financial_analysis


async def test_tutorial() -> None:
    """Test the financial analysis tutorial."""
    model = os.environ.get("MODEL", "gpt-4.1")

    if model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return
    elif model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        return

    try:
        print("🧪 Running financial analysis tutorial test...")
        await demo_financial_analysis()
        print("✅ Tutorial test completed successfully!")

    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tutorial())
