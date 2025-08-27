"""
Demo application for code generation tutorial.
"""

import asyncio
import os

from logillm.providers import create_provider, register_provider

from .generator import LibraryCodeGenerator
from .interactive import InteractiveLearningSession


async def demo_fastapi_generation() -> None:
    """Demonstrate code generation for FastAPI."""

    # Setup LogiLLM provider
    model = os.environ.get("MODEL", "gpt-4.1")

    if model.startswith("gpt"):
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Please set OPENAI_API_KEY environment variable")
            return
        provider = create_provider("openai", model=model)
    elif model.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
            return
        provider = create_provider("anthropic", model=model)
    else:
        raise ValueError(f"Unsupported model: {model}")

    register_provider(provider, set_default=True)

    # Create learning session
    session = InteractiveLearningSession()

    # Explore FastAPI
    await session.explore_library(library_name="fastapi", user_experience="beginner")


async def demo_multiple_libraries() -> None:
    """Demonstrate generation for multiple libraries."""

    model = os.environ.get("MODEL", "gpt-4.1")

    if model.startswith("gpt"):
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Please set OPENAI_API_KEY environment variable")
            return
        provider = create_provider("openai", model=model)
    elif model.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
            return
        provider = create_provider("anthropic", model=model)
    else:
        raise ValueError(f"Unsupported model: {model}")

    register_provider(provider, set_default=True)

    generator = LibraryCodeGenerator()

    libraries = ["requests", "pandas", "click"]
    use_cases = ["basic usage example", "common workflow", "error handling"]

    for library in libraries:
        print(f"\n{'=' * 60}")
        print(f"🔍 Exploring {library.upper()}")
        print(f"{'=' * 60}")

        result = await generator.forward(
            library_name=library,
            use_cases=use_cases,
            user_experience="intermediate",
            fetch_docs=False,  # Use fallback for demo speed
        )

        print(f"\n📊 Analysis for {library}:")
        print(f"   Core concepts: {len(result.analysis.core_concepts)}")
        print(f"   Generated examples: {len(result.code_examples)}")

        # Show first example
        if result.code_examples:
            first_example = list(result.code_examples.values())[0]
            print("\n💻 Sample code:")

            # Handle potential parsing issues
            try:
                code_example = (
                    str(first_example.code_example)
                    if hasattr(first_example, "code_example")
                    else "Code example not available"
                )
                code_preview = (
                    code_example[:200] + "..." if len(code_example) > 200 else code_example
                )
                print(code_preview)
            except (TypeError, AttributeError):
                print("Code example not available")

            print("\n📝 Explanation preview:")
            try:
                explanation = (
                    str(first_example.explanation)
                    if hasattr(first_example, "explanation")
                    else "Explanation not available"
                )
                explanation_preview = (
                    explanation[:150] + "..." if len(explanation) > 150 else explanation
                )
                print(explanation_preview)
            except (TypeError, AttributeError):
                print("Explanation not available")


async def main() -> None:
    """Main demo entry point."""

    print("🚀 LogiLLM Code Generation Tutorial")
    print("=" * 50)

    try:
        # For demo purposes, just run multiple libraries demo
        await demo_multiple_libraries()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
