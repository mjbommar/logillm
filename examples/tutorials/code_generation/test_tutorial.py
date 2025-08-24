"""
Test script for the code generation tutorial.
Run with: uv run --with logillm[openai] --with requests --with beautifulsoup4 python -m examples.tutorials.code_generation.test_tutorial
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.tutorials.code_generation.demo import demo_multiple_libraries


async def test_tutorial() -> None:
    """Test the code generation tutorial."""

    model = os.environ.get("MODEL", "gpt-4.1")

    if model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    elif model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return

    try:
        print("🧪 Running code generation tutorial test...")
        await demo_multiple_libraries()
        print("✅ Tutorial test completed successfully!")

    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tutorial())
