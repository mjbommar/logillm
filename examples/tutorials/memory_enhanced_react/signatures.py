"""LogiLLM signatures for memory-enhanced ReAct agent."""

from logillm.core.signatures import InputField, OutputField, Signature


class MemoryAnalysis(Signature):
    """Analyze user input to extract memorable information."""

    user_input: str = InputField(description="The user's input message")
    user_context: str = InputField(description="Existing context about the user")

    extractable_facts: str = OutputField(
        description='List of facts worth remembering as JSON array: ["fact1", "fact2", "fact3"]'
    )
    memory_type: str = OutputField(
        description="Type of memory: preference, fact, reminder, conversation, experience"
    )
    importance_score: int = OutputField(description="Importance score 1-10")
    suggested_tags: str = OutputField(
        description='Tags for categorizing this memory as JSON array: ["tag1", "tag2", "tag3"]'
    )


class MemoryRetrieval(Signature):
    """Retrieve relevant memories for answering user queries."""

    user_query: str = InputField(description="The user's current query")
    user_id: str = InputField(description="User identifier")

    relevant_memories: str = OutputField(
        description='List of relevant memories to use in response as JSON array: ["memory1", "memory2", "memory3"]'
    )
    search_strategy: str = OutputField(description="How memories were selected")


class PersonalizedResponse(Signature):
    """Generate personalized responses using memory context."""

    user_input: str = InputField(description="Current user input")
    relevant_memories: str = InputField(
        description="Memories relevant to this interaction as JSON array"
    )
    conversation_history: str = InputField(description="Recent conversation turns as JSON array")

    response: str = OutputField(description="Personalized response incorporating memories")
    memory_updates: str = OutputField(
        description='New memories to store from this interaction as JSON array: ["memory1", "memory2"]'
    )
    action_items: str = OutputField(
        description='Any actions or reminders to set as JSON array: ["action1", "action2"]'
    )


class ConversationPlanning(Signature):
    """Plan how to approach a user interaction with memory context."""

    user_input: str = InputField(description="User's message")
    user_profile: str = InputField(description="Summary of what we know about the user")
    available_tools: str = InputField(
        description="Available memory and utility tools as JSON array"
    )

    reasoning: str = OutputField(description="Step-by-step reasoning about how to respond")
    tool_calls: str = OutputField(
        description='Which tools to use and in what order as JSON array: ["tool1", "tool2"]'
    )
    response_strategy: str = OutputField(description="Overall strategy for this response")
