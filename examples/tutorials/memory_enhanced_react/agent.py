"""Memory-Enhanced ReAct Agent implementation using LogiLLM."""

from datetime import datetime
from typing import Any

from logillm.core.modules import Module
from logillm.core.predict import Predict

from .memory_manager import MemoryManager, MemoryType
from .signatures import ConversationPlanning, MemoryAnalysis, MemoryRetrieval, PersonalizedResponse


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
            "store_memory": self._tool_store_memory,
            "search_memories": self._tool_search_memories,
            "set_reminder": self._tool_set_reminder,
            "update_preference": self._tool_update_preference,
            "get_user_profile": self._tool_get_user_profile,
            "get_current_time": self._tool_get_current_time,
        }

    def _parse_json_list(self, text: str) -> list[str]:
        """Parse JSON array from text with fallbacks."""
        import json

        if not text or not isinstance(text, str):
            return []

        text = text.strip()

        # Try JSON array format
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Fallback: split by comma or newlines
        if "," in text:
            return [item.strip() for item in text.split(",") if item.strip()]
        elif "\n" in text:
            return [item.strip() for item in text.split("\n") if item.strip()]

        return [text] if text else []

    # Tool implementations
    def _tool_store_memory(
        self,
        content: str,
        memory_type: str,
        user_id: str = "default_user",
        importance: int = 5,
        tags: str = "",
    ) -> str:
        """Store information in memory."""
        try:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
            memory_type_enum = MemoryType(memory_type)

            memory = self.memory_manager.store_memory(
                user_id, content, memory_type_enum, tag_list, importance
            )
            return f"✅ Stored memory: {content} (ID: {memory.id[:8]}...)"
        except Exception as e:
            return f"❌ Error storing memory: {str(e)}"

    def _tool_search_memories(
        self, query: str, user_id: str = "default_user", memory_type: str = "", limit: int = 5
    ) -> str:
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

    def _tool_set_reminder(
        self, reminder_text: str, date_time: str = "", user_id: str = "default_user"
    ) -> str:
        """Set a reminder for the user."""
        if not date_time:
            date_time = "unspecified time"

        content = f"REMINDER for {date_time}: {reminder_text}"
        return self._tool_store_memory(content, "reminder", user_id, importance=8, tags="reminder")

    def _tool_update_preference(
        self, category: str, value: str, user_id: str = "default_user"
    ) -> str:
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

    async def _analyze_and_store_memories(self, user_input: str, user_id: str) -> list[str]:
        """Analyze input and store relevant memories."""
        context = self.memory_manager.get_conversation_context(user_id)
        user_context = f"User: {context.user_profile.name or user_id}, Preferences: {context.user_profile.preferences}"

        try:
            analysis = await self.memory_analyzer(user_input=user_input, user_context=user_context)

            stored_memories = []
            # Parse string responses to lists
            extractable_facts = self._parse_json_list(analysis.extractable_facts)
            suggested_tags = self._parse_json_list(analysis.suggested_tags)

            if extractable_facts:
                for fact in extractable_facts:
                    if fact.strip():  # Only store non-empty facts
                        self.memory_manager.store_memory(
                            user_id=user_id,
                            content=fact,
                            memory_type=MemoryType(analysis.memory_type),
                            tags=suggested_tags,
                            importance=analysis.importance_score,
                        )
                        stored_memories.append(fact)

            return stored_memories
        except Exception as e:
            print(f"Error analyzing memories: {e}")
            return []

    async def forward(self, user_input: str, user_id: str = "default_user") -> dict[str, Any]:
        """Process user input with memory-aware reasoning."""
        import json

        # Get conversation context
        context = self.memory_manager.get_conversation_context(user_id, user_input)

        # Plan the conversation approach
        user_profile_summary = f"Name: {context.user_profile.name or 'Unknown'}, Memories: {len(context.user_profile.memories)}, Preferences: {context.user_profile.preferences}"

        try:
            plan = await self.conversation_planner(
                user_input=user_input,
                user_profile=user_profile_summary,
                available_tools=json.dumps(list(self.tools.keys())),
            )

            # Retrieve relevant memories
            memory_retrieval = await self.memory_retriever(user_query=user_input, user_id=user_id)

            # Get relevant memories based on analysis
            relevant_memories = []
            retrieved_memories = self._parse_json_list(memory_retrieval.relevant_memories)
            for memory_desc in retrieved_memories:
                memories = self.memory_manager.search_memories(user_id, memory_desc, limit=2)
                relevant_memories.extend([m.content for m in memories])

            # Generate personalized response
            response = await self.response_generator(
                user_input=user_input,
                relevant_memories=json.dumps(relevant_memories),
                conversation_history=json.dumps(context.conversation_history[-5:]),  # Last 5 turns
            )

            # Store new memories from this interaction
            stored_memories = await self._analyze_and_store_memories(user_input, user_id)

            # Execute any tool calls mentioned in the plan
            tool_results = []
            planned_tool_calls = self._parse_json_list(plan.tool_calls)
            for tool_call in planned_tool_calls:
                if any(tool_name in tool_call for tool_name in self.tools.keys()):
                    # This is a simplified tool execution - in a full implementation,
                    # you'd parse the tool call properly
                    tool_results.append(f"Executed: {tool_call}")

            # Update conversation history
            context.conversation_history.extend(
                [f"User: {user_input}", f"Agent: {response.response}"]
            )

            return {
                "user_input": user_input,
                "response": response.response,
                "plan": {
                    "reasoning": plan.reasoning,
                    "strategy": plan.response_strategy,
                    "tool_calls": plan.tool_calls,
                },
                "memories_used": relevant_memories,
                "memories_stored": stored_memories,
                "action_items": self._parse_json_list(response.action_items),
                "tool_results": tool_results,
                "context": {
                    "user_id": user_id,
                    "total_memories": len(context.user_profile.memories),
                    "preferences": context.user_profile.preferences,
                },
            }

        except Exception as e:
            return {
                "user_input": user_input,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "error": str(e),
                "context": {
                    "user_id": user_id,
                    "total_memories": len(context.user_profile.memories),
                },
            }

    async def get_memory_summary(self, user_id: str = "default_user") -> dict[str, Any]:
        """Get a summary of user's memories and profile."""
        context = self.memory_manager.get_conversation_context(user_id)

        # Group memories by type
        memory_counts = {}
        for memory in context.user_profile.memories:
            memory_counts[memory.memory_type.value] = (
                memory_counts.get(memory.memory_type.value, 0) + 1
            )

        return {
            "user_id": user_id,
            "name": context.user_profile.name,
            "total_memories": len(context.user_profile.memories),
            "memory_types": memory_counts,
            "preferences": context.user_profile.preferences,
            "last_interaction": context.user_profile.last_interaction,
            "recent_memories": [m.content for m in context.recent_memories[:5]],
        }
