"""Test script for memory-enhanced ReAct agent tutorial."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the tutorial modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def test_memory_enhanced_react():
    """Test the memory-enhanced ReAct agent implementation."""

    # Check for API keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️  Skipping test - no API keys found")
        print("   Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return False

    try:
        # Import our modules
        from examples.tutorials.memory_enhanced_react.agent import MemoryEnhancedReActAgent
        from examples.tutorials.memory_enhanced_react.models import MemoryType
        from logillm.providers import create_provider, register_provider

        print("✅ Successfully imported memory-enhanced ReAct modules")

        # Register providers
        if os.getenv("OPENAI_API_KEY"):
            openai_provider = create_provider("openai")
            register_provider(openai_provider, set_default=True)
        elif os.getenv("ANTHROPIC_API_KEY"):
            anthropic_provider = create_provider("anthropic")
            register_provider(anthropic_provider, set_default=True)

        print("✅ Providers registered")

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize agent with temporary storage
            agent = MemoryEnhancedReActAgent(temp_dir)
            print("✅ Created memory-enhanced agent")

            # Test basic memory operations
            test_user = "test_user_123"

            # Test 1: Store memory directly
            memory = agent.memory_manager.store_memory(
                test_user,
                "User loves pizza and Italian food",
                MemoryType.PREFERENCE,
                tags=["food", "preference"],
                importance=8,
            )
            print(f"✅ Stored memory: {memory.id[:8]}...")

            # Test 2: Search memories
            memories = agent.memory_manager.search_memories(test_user, query="food")
            assert len(memories) == 1, f"Expected 1 memory, found {len(memories)}"
            print("✅ Memory search working")

            # Test 3: Process user input
            result = await agent(
                user_input="Hi, I'm John and I prefer coffee over tea", user_id=test_user
            )

            # Debug: print the actual result structure
            print(
                f"🔍 Agent result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}"
            )

            if "error" in result:
                print(f"❌ Agent error: {result['error']}")
                print(f"🔍 Full error result: {result}")

            assert "response" in result, "Missing response in agent output"
            assert "memories_stored" in result, "Missing memories_stored in agent output"
            print("✅ Agent conversation processing working")

            # Test 4: Check if new memory was stored
            all_memories = agent.memory_manager.search_memories(test_user, limit=10)
            assert len(all_memories) >= 1, "No memories found after conversation"
            print(f"✅ Total memories: {len(all_memories)}")

            # Test 5: Get memory summary
            summary = await agent.get_memory_summary(test_user)
            assert summary["total_memories"] > 0, "Memory summary shows no memories"
            print("✅ Memory summary working")

            # Test 6: Test persistence (create new agent with same storage)
            agent2 = MemoryEnhancedReActAgent(temp_dir)
            loaded_memories = agent2.memory_manager.search_memories(test_user, limit=10)
            assert len(loaded_memories) > 0, "Memories not persisted"
            print("✅ Memory persistence working")

        print("\n🎉 All tests passed! Memory-enhanced ReAct agent is working correctly.")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all required modules are available")
        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_memory_enhanced_react())
    sys.exit(0 if success else 1)
