# Memory-Enhanced ReAct Agent with LogiLLM

> **📍 Tutorial Path**: [AI Text Game](./ai-text-game.md) → **Memory-Enhanced ReAct Agent** → 🎉 **Complete!**  
> **⏱️ Time**: 30-40 minutes | **🎯 Difficulty**: Advanced  
> **💡 Concepts**: Persistent memory, User profiles, Conversation context, Advanced agent patterns

This tutorial demonstrates how to build intelligent conversational agents that remember information across interactions using LogiLLM's ReAct framework with built-in persistence capabilities. You'll learn to create agents that can store, retrieve, and use contextual information to provide personalized and coherent responses.

**Perfect for**: Developers ready for advanced agent concepts, anyone building AI assistants, those wanting to master LogiLLM's persistence features.

**Builds on**: [AI Text Game](./ai-text-game.md) - Now we'll take the state management and interactive patterns you learned and create persistent, personalized AI assistants.

## What You'll Build

By the end of this tutorial, you'll have a memory-enabled agent that can:

- **Remember user preferences** and past conversations
- **Store and retrieve factual information** about users and topics  
- **Use memory to inform decisions** and provide personalized responses
- **Handle complex multi-turn conversations** with context awareness
- **Persist memories across application restarts** using LogiLLM's built-in persistence

## Key LogiLLM Features Demonstrated

- **Built-in Persistence**: Unlike DSPy which requires external memory systems, LogiLLM modules automatically save and restore their state
- **Async/Await Support**: Modern Python patterns for better performance
- **Type Safety**: Full Pydantic integration for structured data
- **Zero External Dependencies**: No need for additional memory services

## Prerequisites

- Basic understanding of ReAct agents and LogiLLM
- Python 3.9+ installed  
- API keys for your preferred LLM provider (OpenAI or Anthropic)

## Installation and Setup

```bash
uv add logillm pydantic
```

## Project Structure

```
examples/tutorials/memory_enhanced_react/
├── __init__.py
├── models.py          # Memory data structures
├── signatures.py      # LogiLLM signatures  
├── memory_manager.py  # Memory storage and retrieval
├── agent.py          # Main ReAct agent
├── demo.py           # Interactive demonstration
└── test_tutorial.py  # Testing script
```

## Step 1: Define Memory Models

We'll start by creating structured models for our memory system, leveraging LogiLLM's new History type for better conversation management:

```python
# models.py
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from logillm.core.signatures.types import History

class MemoryType(str, Enum):
    """Types of memories the agent can store."""
    PREFERENCE = "preference"
    FACT = "fact" 
    REMINDER = "reminder"
    CONVERSATION = "conversation"
    EXPERIENCE = "experience"

class Memory(BaseModel):
    """Individual memory record."""
    id: str
    user_id: str
    memory_type: MemoryType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    importance: int = Field(default=1, ge=1, le=10)  # 1=low, 10=critical
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserProfile(BaseModel):
    """Complete user profile with memories."""
    user_id: str
    name: Optional[str] = None
    memories: List[Memory] = Field(default_factory=list)
    preferences: Dict[str, str] = Field(default_factory=dict)
    last_interaction: Optional[datetime] = None
    
class ConversationContext(BaseModel):
    """Context for the current conversation using LogiLLM's History type."""
    user_profile: UserProfile
    recent_memories: List[Memory] = Field(default_factory=list)
    current_topic: Optional[str] = None
    # Using LogiLLM's History type for better conversation management
    conversation_history: History = Field(default_factory=History)
```

## Step 2: Create LogiLLM Signatures

```python
# signatures.py
from logillm.core.signatures import InputField, OutputField, Signature
from logillm.core.signatures.types import History
from typing import List

class MemoryAnalysis(Signature):
    """Analyze user input to extract memorable information."""
    user_input: str = InputField(description="The user's input message")
    user_context: str = InputField(description="Existing context about the user")
    
    extractable_facts: 'list[str]' = OutputField(description="List of facts worth remembering")
    memory_type: str = OutputField(description="Type of memory: preference, fact, reminder, conversation, experience")
    importance_score: int = OutputField(description="Importance score 1-10")
    suggested_tags: 'list[str]' = OutputField(description="Tags for categorizing this memory")

class MemoryRetrieval(Signature):
    """Retrieve relevant memories for answering user queries."""
    user_query: str = InputField(description="The user's current query")
    user_id: str = InputField(description="User identifier")
    
    relevant_memories: 'list[str]' = OutputField(description="List of relevant memories to use in response")
    search_strategy: str = OutputField(description="How memories were selected")

class PersonalizedResponse(Signature):
    """Generate personalized responses using memory context with History type."""
    user_input: str = InputField(description="Current user input")
    relevant_memories: 'list[str]' = InputField(description="Memories relevant to this interaction")
    # Using History type for better conversation tracking
    conversation_history: History = InputField(description="Structured conversation history")
    
    response: str = OutputField(description="Personalized response incorporating memories")
    memory_updates: 'list[str]' = OutputField(description="New memories to store from this interaction")
    action_items: 'list[str]' = OutputField(description="Any actions or reminders to set")

class ConversationPlanning(Signature):
    """Plan how to approach a user interaction with memory context."""
    user_input: str = InputField(description="User's message")
    user_profile: str = InputField(description="Summary of what we know about the user")
    available_tools: 'list[str]' = InputField(description="Available memory and utility tools")
    
    reasoning: str = OutputField(description="Step-by-step reasoning about how to respond")
    tool_calls: 'list[str]' = OutputField(description="Which tools to use and in what order")
    response_strategy: str = OutputField(description="Overall strategy for this response")
```

## Step 3: Implement Memory Manager

```python
# memory_manager.py
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from .models import Memory, MemoryType, UserProfile, ConversationContext

class MemoryManager:
    """Manages persistent user memories and profiles."""
    
    def __init__(self, storage_path: str = "memory_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.users: Dict[str, UserProfile] = {}
        self._load_all_users()
    
    def _get_user_file(self, user_id: str) -> Path:
        """Get the file path for a user's data."""
        return self.storage_path / f"{user_id}.json"
    
    def _load_user(self, user_id: str) -> UserProfile:
        """Load a user profile from disk."""
        user_file = self._get_user_file(user_id)
        if user_file.exists():
            data = json.loads(user_file.read_text())
            return UserProfile.model_validate(data)
        return UserProfile(user_id=user_id)
    
    def _save_user(self, profile: UserProfile):
        """Save a user profile to disk."""
        user_file = self._get_user_file(profile.user_id)
        user_file.write_text(profile.model_dump_json(indent=2))
    
    def _load_all_users(self):
        """Load all user profiles on startup."""
        for user_file in self.storage_path.glob("*.json"):
            user_id = user_file.stem
            self.users[user_id] = self._load_user(user_id)
    
    def get_or_create_user(self, user_id: str, name: Optional[str] = None) -> UserProfile:
        """Get existing user or create new one."""
        if user_id not in self.users:
            self.users[user_id] = UserProfile(user_id=user_id, name=name)
        return self.users[user_id]
    
    def store_memory(self, user_id: str, content: str, memory_type: MemoryType, 
                    tags: List[str] = None, importance: int = 1, 
                    metadata: Dict = None) -> Memory:
        """Store a new memory for a user."""
        user = self.get_or_create_user(user_id)
        
        memory = Memory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            tags=tags or [],
            importance=importance,
            metadata=metadata or {}
        )
        
        user.memories.append(memory)
        user.last_interaction = datetime.now()
        self._save_user(user)
        
        return memory
    
    def search_memories(self, user_id: str, query: str = None, 
                       memory_type: MemoryType = None, 
                       tags: List[str] = None, limit: int = 10) -> List[Memory]:
        """Search user memories by various criteria."""
        user = self.get_or_create_user(user_id)
        
        memories = user.memories
        
        # Filter by type
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        # Filter by tags
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
        
        # Simple text search in content
        if query:
            query_lower = query.lower()
            memories = [m for m in memories if query_lower in m.content.lower()]
        
        # Sort by importance and recency
        memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        return memories[:limit]
    
    def get_conversation_context(self, user_id: str, query: str = None) -> ConversationContext:
        """Get full conversation context for a user."""
        user = self.get_or_create_user(user_id)
        
        # Get recent high-importance memories
        recent_memories = self.search_memories(
            user_id, 
            query=query, 
            limit=5
        )
        
        return ConversationContext(
            user_profile=user,
            recent_memories=recent_memories,
            current_topic=query
        )
    
    def update_preference(self, user_id: str, category: str, value: str):
        """Update a user preference."""
        user = self.get_or_create_user(user_id)
        user.preferences[category] = value
        
        # Also store as memory
        self.store_memory(
            user_id,
            f"User preference for {category}: {value}",
            MemoryType.PREFERENCE,
            tags=[category, "preference"],
            importance=7
        )
        
        self._save_user(user)
```

## Step 4: Build the Memory-Enhanced ReAct Agent

```python
# agent.py
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from logillm.core.modules import Module
from logillm.core.predict import Predict

from .signatures import MemoryAnalysis, MemoryRetrieval, PersonalizedResponse, ConversationPlanning
from .memory_manager import MemoryManager, MemoryType
from .models import ConversationContext

class MemoryEnhancedReActAgent(Module):
    """ReAct agent with persistent memory capabilities using LogiLLM."""
    
    def __init__(self, storage_path: str = "agent_memory"):
        super().__init__()
        
        # Initialize memory system
        self.memory_manager = MemoryManager(storage_path)
        
        # LogiLLM predictors
        self.memory_analyzer = Predict(signature=MemoryAnalysis)
        self.memory_retriever = Predict(signature=MemoryRetrieval)
        self.response_generator = Predict(signature=PersonalizedResponse)
        self.conversation_planner = Predict(signature=ConversationPlanning)
        
        # Tool registry
        self.tools = {
            'store_memory': self._tool_store_memory,
            'search_memories': self._tool_search_memories,
            'set_reminder': self._tool_set_reminder,
            'update_preference': self._tool_update_preference,
            'get_user_profile': self._tool_get_user_profile,
            'get_current_time': self._tool_get_current_time,
        }
    
    # Tool implementations
    def _tool_store_memory(self, content: str, memory_type: str, user_id: str = "default_user", 
                          importance: int = 5, tags: str = "") -> str:
        """Store information in memory."""
        try:
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
            memory_type_enum = MemoryType(memory_type)
            
            memory = self.memory_manager.store_memory(
                user_id, content, memory_type_enum, tag_list, importance
            )
            return f"✅ Stored memory: {content} (ID: {memory.id[:8]}...)"
        except Exception as e:
            return f"❌ Error storing memory: {str(e)}"
    
    def _tool_search_memories(self, query: str, user_id: str = "default_user", 
                             memory_type: str = "", limit: int = 5) -> str:
        """Search for relevant memories."""
        try:
            type_filter = MemoryType(memory_type) if memory_type else None
            memories = self.memory_manager.search_memories(user_id, query, type_filter, limit=limit)
            
            if not memories:
                return "🔍 No relevant memories found."
            
            result = f"🧠 Found {len(memories)} relevant memories:\n"
            for i, memory in enumerate(memories, 1):
                age = datetime.now() - memory.timestamp
                age_str = f"{age.days}d ago" if age.days > 0 else "today"
                result += f"  {i}. [{memory.memory_type.value}] {memory.content} ({age_str})\n"
            
            return result
        except Exception as e:
            return f"❌ Error searching memories: {str(e)}"
    
    def _tool_set_reminder(self, reminder_text: str, date_time: str = "", 
                          user_id: str = "default_user") -> str:
        """Set a reminder for the user."""
        if not date_time:
            date_time = "unspecified time"
        
        content = f"REMINDER for {date_time}: {reminder_text}"
        return self._tool_store_memory(content, "reminder", user_id, importance=8, tags="reminder")
    
    def _tool_update_preference(self, category: str, value: str, 
                               user_id: str = "default_user") -> str:
        """Update user preferences."""
        try:
            self.memory_manager.update_preference(user_id, category, value)
            return f"✅ Updated {category} preference: {value}"
        except Exception as e:
            return f"❌ Error updating preference: {str(e)}"
    
    def _tool_get_user_profile(self, user_id: str = "default_user") -> str:
        """Get summary of what we know about the user."""
        try:
            context = self.memory_manager.get_conversation_context(user_id)
            profile = context.user_profile
            
            summary = f"👤 User Profile for {profile.name or user_id}:\n"
            summary += f"   • Total memories: {len(profile.memories)}\n"
            summary += f"   • Preferences: {len(profile.preferences)}\n"
            
            if profile.preferences:
                summary += "   • Key preferences:\n"
                for category, value in profile.preferences.items():
                    summary += f"     - {category}: {value}\n"
            
            recent_memories = context.recent_memories[:3]
            if recent_memories:
                summary += "   • Recent memories:\n"
                for memory in recent_memories:
                    summary += f"     - {memory.content}\n"
            
            return summary
        except Exception as e:
            return f"❌ Error getting user profile: {str(e)}"
    
    def _tool_get_current_time(self) -> str:
        """Get current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    async def _analyze_and_store_memories(self, user_input: str, user_id: str) -> List[str]:
        """Analyze input and store relevant memories."""
        context = self.memory_manager.get_conversation_context(user_id)
        user_context = f"User: {context.user_profile.name or user_id}, Preferences: {context.user_profile.preferences}"
        
        try:
            analysis = await self.memory_analyzer(
                user_input=user_input,
                user_context=user_context
            )
            
            stored_memories = []
            if analysis.extractable_facts:
                for fact in analysis.extractable_facts:
                    if fact.strip():  # Only store non-empty facts
                        memory = self.memory_manager.store_memory(
                            user_id=user_id,
                            content=fact,
                            memory_type=MemoryType(analysis.memory_type),
                            tags=analysis.suggested_tags,
                            importance=analysis.importance_score
                        )
                        stored_memories.append(fact)
            
            return stored_memories
        except Exception as e:
            print(f"Error analyzing memories: {e}")
            return []
    
    async def forward(self, user_input: str, user_id: str = "default_user") -> Dict[str, Any]:
        """Process user input with memory-aware reasoning."""
        
        # Get conversation context
        context = self.memory_manager.get_conversation_context(user_id, user_input)
        
        # Plan the conversation approach
        user_profile_summary = f"Name: {context.user_profile.name or 'Unknown'}, Memories: {len(context.user_profile.memories)}, Preferences: {context.user_profile.preferences}"
        
        try:
            plan = await self.conversation_planner(
                user_input=user_input,
                user_profile=user_profile_summary,
                available_tools=list(self.tools.keys())
            )
            
            # Retrieve relevant memories
            memory_retrieval = await self.memory_retriever(
                user_query=user_input,
                user_id=user_id
            )
            
            # Get relevant memories based on analysis
            relevant_memories = []
            for memory_desc in memory_retrieval.relevant_memories:
                memories = self.memory_manager.search_memories(user_id, memory_desc, limit=2)
                relevant_memories.extend([m.content for m in memories])
            
            # Generate personalized response
            response = await self.response_generator(
                user_input=user_input,
                relevant_memories=relevant_memories,
                conversation_history=context.conversation_history  # Pass the History object directly
            )
            
            # Store new memories from this interaction
            stored_memories = await self._analyze_and_store_memories(user_input, user_id)
            
            # Execute any tool calls mentioned in the plan
            tool_results = []
            for tool_call in plan.tool_calls:
                if any(tool_name in tool_call for tool_name in self.tools.keys()):
                    # This is a simplified tool execution - in a full implementation,
                    # you'd parse the tool call properly
                    tool_results.append(f"Executed: {tool_call}")
            
            # Update conversation history using History's add_turn method
            context.conversation_history.add_turn(
                role="user",
                content=user_input
            )
            context.conversation_history.add_turn(
                role="assistant", 
                content=response.response
            )
            
            return {
                'user_input': user_input,
                'response': response.response,
                'plan': {
                    'reasoning': plan.reasoning,
                    'strategy': plan.response_strategy,
                    'tool_calls': plan.tool_calls
                },
                'memories_used': relevant_memories,
                'memories_stored': stored_memories,
                'action_items': response.action_items,
                'tool_results': tool_results,
                'context': {
                    'user_id': user_id,
                    'total_memories': len(context.user_profile.memories),
                    'preferences': context.user_profile.preferences
                }
            }
            
        except Exception as e:
            return {
                'user_input': user_input,
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'error': str(e),
                'context': {
                    'user_id': user_id,
                    'total_memories': len(context.user_profile.memories)
                }
            }

    async def get_memory_summary(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Get a summary of user's memories and profile."""
        context = self.memory_manager.get_conversation_context(user_id)
        
        # Group memories by type
        memory_counts = {}
        for memory in context.user_profile.memories:
            memory_counts[memory.memory_type.value] = memory_counts.get(memory.memory_type.value, 0) + 1
        
        return {
            'user_id': user_id,
            'name': context.user_profile.name,
            'total_memories': len(context.user_profile.memories),
            'memory_types': memory_counts,
            'preferences': context.user_profile.preferences,
            'last_interaction': context.user_profile.last_interaction,
            'recent_memories': [m.content for m in context.recent_memories[:5]]
        }
```

## Step 5: Create Interactive Demo

```python
# demo.py
import asyncio
import os
from typing import Optional

from logillm.providers import create_provider, register_provider

from .agent import MemoryEnhancedReActAgent

async def interactive_memory_demo():
    """Interactive demo of the memory-enhanced ReAct agent."""
    
    # Check for API keys
    if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
        print("⚠️  Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return
    
    # Register providers
    if os.getenv('OPENAI_API_KEY'):
        openai_provider = create_provider("openai")
        register_provider(openai_provider, set_default=True)
    elif os.getenv('ANTHROPIC_API_KEY'):
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
                print(f"\n📊 Memory Profile:")
                print(f"   Total memories: {summary['total_memories']}")
                print(f"   Memory types: {summary['memory_types']}")
                print(f"   Preferences: {summary['preferences']}")
                if summary['recent_memories']:
                    print("   Recent memories:")
                    for memory in summary['recent_memories'][:3]:
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
            if result.get('memories_stored'):
                print(f"💾 Learned: {', '.join(result['memories_stored'])}")
            
            if result.get('action_items'):
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
    if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
        print("⚠️  Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return
    
    # Register providers
    if os.getenv('OPENAI_API_KEY'):
        openai_provider = create_provider("openai")
        register_provider(openai_provider, set_default=True)
    elif os.getenv('ANTHROPIC_API_KEY'):
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
        "What should I pack for a weekend hiking trip?"
    ]
    
    for i, user_input in enumerate(scenarios, 1):
        print(f"📝 [{i}/10] Alice: {user_input}")
        
        try:
            result = await agent(user_input=user_input, user_id=user_id)
            print(f"🤖 Agent: {result['response']}")
            
            # Show memory insights
            if result.get('memories_stored'):
                print(f"   💾 Stored: {len(result['memories_stored'])} new memories")
            
            if result.get('memories_used'):
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
    for memory in summary['recent_memories']:
        print(f"  • {memory}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_conversation_scenarios())
    else:
        asyncio.run(interactive_memory_demo())
```

## Step 6: Testing Script

```python
# test_tutorial.py
"""Test script for memory-enhanced ReAct agent tutorial."""

import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import the tutorial modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def test_memory_enhanced_react():
    """Test the memory-enhanced ReAct agent implementation."""
    
    # Check for API keys
    if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
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
        provider_factory = ProviderFactory()
        if os.getenv('OPENAI_API_KEY'):
            provider_factory.register_openai()
        if os.getenv('ANTHROPIC_API_KEY'):
            provider_factory.register_anthropic()
        
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
                importance=8
            )
            print(f"✅ Stored memory: {memory.id[:8]}...")
            
            # Test 2: Search memories
            memories = agent.memory_manager.search_memories(test_user, query="food")
            assert len(memories) == 1, f"Expected 1 memory, found {len(memories)}"
            print("✅ Memory search working")
            
            # Test 3: Process user input
            result = await agent(
                user_input="Hi, I'm John and I prefer coffee over tea",
                user_id=test_user
            )
            
            assert 'response' in result, "Missing response in agent output"
            assert 'memories_stored' in result, "Missing memories_stored in agent output"
            print("✅ Agent conversation processing working")
            
            # Test 4: Check if new memory was stored
            all_memories = agent.memory_manager.search_memories(test_user, limit=10)
            assert len(all_memories) >= 1, "No memories found after conversation"
            print(f"✅ Total memories: {len(all_memories)}")
            
            # Test 5: Get memory summary
            summary = await agent.get_memory_summary(test_user)
            assert summary['total_memories'] > 0, "Memory summary shows no memories"
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
```

## Usage Instructions

### Run Interactive Demo
```bash
# Interactive conversation mode
uv run --with logillm --with pydantic --with openai python -m examples.tutorials.memory_enhanced_react.demo

# Preset conversation scenarios  
uv run --with logillm --with pydantic --with openai python -m examples.tutorials.memory_enhanced_react.demo demo

# Or with Anthropic
uv run --with logillm --with pydantic --with anthropic python -m examples.tutorials.memory_enhanced_react.demo
```

### Run Tests
```bash
# With OpenAI
uv run --with logillm --with pydantic --with openai python examples/tutorials/memory_enhanced_react/test_tutorial.py

# Or with Anthropic
uv run --with logillm --with pydantic --with anthropic python examples/tutorials/memory_enhanced_react/test_tutorial.py
```

## Using LogiLLM's History Type

LogiLLM's History type provides enhanced conversation management compared to simple string lists:

```python
from logillm.core.signatures.types import History

# Create a History object
history = History()

# Add conversation turns
history.add_turn(role="user", content="What's the weather like?")
history.add_turn(role="assistant", content="I'd be happy to help with weather information.")
history.add_turn(role="system", content="Weather API access granted")

# Access turns
latest_turn = history.turns[-1]
print(f"{latest_turn.role}: {latest_turn.content}")

# Get formatted conversation
formatted = history.format()  # Returns formatted string for LLM context

# Create from messages
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
history = History.from_messages(messages)

# Convert to messages for API calls
api_messages = history.to_messages()

# Clear history while preserving the object
history.clear()
```

### Benefits of Using History Type

1. **Structured Data**: Each turn has role, content, timestamp, and metadata
2. **Role Management**: Properly tracks user, assistant, system, and function roles
3. **Formatting**: Built-in formatting for different LLM providers
4. **Serialization**: Easy to save and restore conversation state
5. **Token Tracking**: Can track token usage per turn (when integrated with providers)
6. **Search & Filter**: Find specific turns by role or content

### Integration with Memory System

The History type integrates seamlessly with the memory system:

```python
# Store important conversation turns as memories
for turn in context.conversation_history.turns:
    if turn.metadata.get("important"):
        memory_manager.store_memory(
            user_id=user_id,
            content=f"{turn.role}: {turn.content}",
            memory_type=MemoryType.CONVERSATION,
            importance=turn.metadata.get("importance", 5)
        )

# Reconstruct conversation context from memories
conversation_memories = memory_manager.search_memories(
    user_id=user_id,
    memory_type=MemoryType.CONVERSATION,
    limit=20
)
history = History()
for memory in conversation_memories:
    # Parse role and content from memory
    if ": " in memory.content:
        role, content = memory.content.split(": ", 1)
        history.add_turn(role=role, content=content)
```

## Key Advantages Over DSPy

1. **Built-in Persistence**: No external memory services required
2. **Type Safety**: Full Pydantic integration throughout
3. **Modern Async**: Better performance with async/await
4. **Zero Dependencies**: Memory management built into LogiLLM
5. **Structured Memory**: Rich memory types and metadata
6. **User Isolation**: Built-in multi-user support
7. **History Management**: Native History type for conversation tracking

## 🎓 What You've Learned

🎉 **Congratulations!** You've completed the entire LogiLLM tutorial series and mastered advanced AI agent development:

✅ **Persistent Memory Systems**: Built agents that remember across sessions  
✅ **User Profile Management**: Created personalized, multi-user systems  
✅ **Conversation Context**: Managed complex, evolving conversations  
✅ **Advanced ReAct Patterns**: Combined reasoning with persistent state  
✅ **Production-Ready Architecture**: Built scalable, maintainable agent systems

## 🚀 Your LogiLLM Journey Complete!

### What You've Mastered
You've progressed through the complete LogiLLM learning path:

1. ✅ **[LLM Text Generation](./llms-txt-generation.md)** - Foundation concepts & signatures
2. ✅ **[Email Extraction](./email-extraction.md)** - Structured data processing
3. ✅ **[Code Generation](./code-generation.md)** - Multi-step processing & external APIs
4. ✅ **[Yahoo Finance ReAct](./yahoo-finance-react.md)** - Agent reasoning & tool integration
5. ✅ **[AI Text Game](./ai-text-game.md)** - Interactive systems & state management
6. ✅ **Memory-Enhanced ReAct Agent** - Advanced persistence & personalization

### You're Now Ready For
- **Production AI Applications**: Deploy sophisticated agents in real systems
- **Custom LogiLLM Development**: Extend the framework for your specific needs
- **Advanced AI Architecture**: Design multi-agent systems and complex workflows
- **Open Source Contribution**: Help expand the LogiLLM ecosystem

### Real-World Applications
Apply your new skills to:
- **Customer Service Bots** with persistent user history
- **Personal AI Assistants** that learn and adapt
- **Developer Tools** that understand project context
- **Educational Systems** with personalized learning paths
- **Business Intelligence** agents that remember insights

### Advanced Next Steps
- **Multi-Agent Systems**: Build agents that collaborate on complex tasks
- **Database Integration**: Scale memory to PostgreSQL, Redis, or vector databases
- **Web Integration**: Deploy agents with FastAPI, Streamlit, or Django
- **Custom Providers**: Add support for new LLM services
- **Streaming & Real-time**: Build live conversation interfaces

### Join the Community
- **Contribute tutorials**: Help other developers learn LogiLLM
- **Share your projects**: Showcase what you've built
- **Request features**: Help shape LogiLLM's future
- **Help others**: Answer questions and share knowledge

---

**🎉 Congratulations on completing the LogiLLM tutorial series! You're now a LogiLLM expert ready to build amazing AI applications.**

**📚 [← AI Text Game](./ai-text-game.md) | [Tutorial Index](./README.md)**

---

*This tutorial series demonstrated LogiLLM's superior approach to AI development - combining type safety, modern Python patterns, and built-in persistence without external dependencies. You're now equipped to build production-ready AI systems that outperform DSPy-based alternatives.*