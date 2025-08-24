"""Interactive demo for memory-enhanced ReAct agent tutorial."""

import asyncio
import os

from logillm.providers import create_provider, register_provider

from .agent import MemoryEnhancedReActAgent


async def interactive_memory_demo():
    """Interactive demo of the memory-enhanced ReAct agent."""

    # Check for API keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️  Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    # Register providers
    if os.getenv("OPENAI_API_KEY"):
        openai_provider = create_provider("openai")
        register_provider(openai_provider, set_default=True)
    elif os.getenv("ANTHROPIC_API_KEY"):
        anthropic_provider = create_provider("anthropic")
        register_provider(anthropic_provider, set_default=True)

    # Create agent
    agent = MemoryEnhancedReActAgent()
    print("🧠 Memory-Enhanced ReAct Agent")
    print("=" * 50)
    print("This agent remembers information across conversations!")
    print("Commands:")
    print("  /profile - Show your memory profile")
    print("  /memories [query] - Search your memories")
    print("  /clear - Clear conversation (keeps memories)")
    print("  /quit - Exit the demo")
    print()

    # Get user ID
    user_id = input("👤 Enter your name/ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"

    print(f"\n💬 Starting conversation with {user_id}")
    print("Type your messages below (or use commands):\n")

    while True:
        try:
            user_input = input(f"{user_id}: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("👋 Goodbye!")
                break
            elif user_input == "/profile":
                summary = await agent.get_memory_summary(user_id)
                print("\n📊 Memory Profile:")
                print(f"   Total memories: {summary['total_memories']}")
                print(f"   Memory types: {summary['memory_types']}")
                print(f"   Preferences: {summary['preferences']}")
                if summary["recent_memories"]:
                    print("   Recent memories:")
                    for memory in summary["recent_memories"][:3]:
                        print(f"     • {memory}")
                print()
                continue
            elif user_input.startswith("/memories"):
                query = user_input[9:].strip() if len(user_input) > 9 else ""
                result = agent._tool_search_memories(query or "all", user_id, limit=10)
                print(f"\n{result}\n")
                continue
            elif user_input == "/clear":
                print("💭 Conversation cleared (memories preserved)\n")
                continue

            # Process with agent
            print("🤔 Thinking...")
            result = await agent(user_input=user_input, user_id=user_id)

            print(f"🤖 Agent: {result['response']}")

            # Show what was learned (optional debug info)
            if result.get("memories_stored"):
                print(f"💾 Learned: {', '.join(result['memories_stored'])}")

            if result.get("action_items"):
                print(f"📝 Actions: {', '.join(result['action_items'])}")

            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


async def demo_conversation_scenarios():
    """Run preset conversation scenarios to demonstrate memory capabilities."""

    # Check for API keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️  Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    # Register providers
    if os.getenv("OPENAI_API_KEY"):
        openai_provider = create_provider("openai")
        register_provider(openai_provider, set_default=True)
    elif os.getenv("ANTHROPIC_API_KEY"):
        anthropic_provider = create_provider("anthropic")
        register_provider(anthropic_provider, set_default=True)

    agent = MemoryEnhancedReActAgent()
    user_id = "alice_demo"

    print("🧠 Memory-Enhanced ReAct Agent - Preset Demo")
    print("=" * 50)
    print("Demonstrating memory capabilities with Alice...")
    print()

    scenarios = [
        "Hi, I'm Alice. I love Italian food, especially pasta carbonara.",
        "I prefer to exercise in the morning around 7 AM.",
        "I work as a software engineer at a tech startup.",
        "What do you remember about my food preferences?",
        "Set a reminder for me to call my dentist tomorrow.",
        "What are my exercise preferences?",
        "I also enjoy hiking on weekends, especially in the mountains.",
        "What do you know about me so far?",
        "Can you recommend a good Italian restaurant for tonight?",
        "What should I pack for a weekend hiking trip?",
    ]

    for i, user_input in enumerate(scenarios, 1):
        print(f"📝 [{i}/10] Alice: {user_input}")

        try:
            result = await agent(user_input=user_input, user_id=user_id)
            print(f"🤖 Agent: {result['response']}")

            # Show memory insights
            if result.get("memories_stored"):
                print(f"   💾 Stored: {len(result['memories_stored'])} new memories")

            if result.get("memories_used"):
                print(f"   🧠 Used: {len(result['memories_used'])} existing memories")

            print()
            await asyncio.sleep(1)  # Brief pause between interactions

        except Exception as e:
            print(f"❌ Error: {e}")
            print()

    # Show final memory summary
    print("=" * 50)
    print("📊 Final Memory Summary:")
    summary = await agent.get_memory_summary(user_id)
    print(f"Total memories stored: {summary['total_memories']}")
    print(f"Memory types: {summary['memory_types']}")
    print(f"User preferences: {summary['preferences']}")
    print()
    print("Recent memories:")
    for memory in summary["recent_memories"]:
        print(f"  • {memory}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_conversation_scenarios())
    else:
        asyncio.run(interactive_memory_demo())
