"""
Test script for the llms.txt generation tutorial.
Run with: uv run --with logillm[openai] --with requests python -m examples.tutorials.llms_txt_generation.test_tutorial
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.tutorials.llms_txt_generation.llms_txt_generator import generate_llms_txt_for_repo


async def test_tutorial() -> None:
    """Test the llms.txt generation tutorial."""

    # Ensure API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return

    # Test with a small public repository
    test_repo = "https://github.com/octocat/Hello-World"

    try:
        await generate_llms_txt_for_repo(
            repo_url=test_repo, output_path="test_llms.txt", model="gpt-4.1"
        )
        print("✅ Tutorial test completed successfully!")

        # Verify output file was created
        if Path("test_llms.txt").exists():
            print("📄 Generated file exists and contains:")
            content = Path("test_llms.txt").read_text()
            print(f"   {len(content)} characters")
            print(f"   First line: {content.split(chr(10))[0] if content else 'Empty'}")
        else:
            print("❌ Output file was not created")

    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tutorial())
