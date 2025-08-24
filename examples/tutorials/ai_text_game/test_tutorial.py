"""Test for AI text game tutorial."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.tutorials.ai_text_game.demo import simple_text_adventure_demo


async def test_tutorial() -> None:
    """Test the AI text game tutorial."""
    model = os.environ.get("MODEL", "gpt-4.1")

    if model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return
    elif model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        return

    try:
        print("🧪 Running AI text game tutorial test...")
        await simple_text_adventure_demo()
        print("✅ Tutorial test completed successfully!")

    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tutorial())
