"""
Test script for the email extraction tutorial.
Run with: uv run --with logillm[openai] --with pydantic python -m examples.tutorials.email_extraction.test_tutorial
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.tutorials.email_extraction.demo import run_email_processing_demo


async def test_tutorial() -> None:
    """Test the email extraction tutorial."""

    # Check for required environment variables
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
        print("🧪 Running email processing tutorial test...")
        await run_email_processing_demo()
        print("✅ Tutorial test completed successfully!")

    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tutorial())
