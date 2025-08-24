"""Demo for AI text adventure game."""

import asyncio
import os

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider


async def simple_text_adventure_demo() -> None:
    """Simple demonstration of AI text adventure concepts."""

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

    print("🎮 LogiLLM Text Adventure Demo")
    print("=" * 40)

    # Simple story generator
    story_generator = Predict("player_action, current_scene -> new_scene, available_actions: list")

    current_scene = (
        "You wake up in a mysterious forest clearing. Sunlight filters through ancient trees."
    )
    actions = ["explore north", "examine surroundings", "look for supplies"]

    print(f"📖 {current_scene}")
    print(f"\n🎯 Available actions: {', '.join(actions)}")

    # Simulate a few story beats
    for i in range(3):
        print(f"\n--- Turn {i + 1} ---")

        # Use first action for demo
        chosen_action = actions[0] if actions else "wait"
        print(f"⚡ You choose: {chosen_action}")

        try:
            result = await story_generator(player_action=chosen_action, current_scene=current_scene)

            current_scene = result.new_scene
            actions = (
                result.available_actions
                if hasattr(result, "available_actions") and result.available_actions
                else ["continue exploring", "rest", "examine area"]
            )

            print(f"\n📜 {current_scene}")
            print(f"🎯 New actions: {', '.join(actions[:3])}")

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    print("\n🎭 Demo complete!")


async def main() -> None:
    await simple_text_adventure_demo()


if __name__ == "__main__":
    asyncio.run(main())
