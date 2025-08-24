"""Memory models for the Memory-Enhanced ReAct Agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=1, ge=1, le=10)  # 1=low, 10=critical
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserProfile(BaseModel):
    """Complete user profile with memories."""

    user_id: str
    name: Optional[str] = None
    memories: list[Memory] = Field(default_factory=list)
    preferences: dict[str, str] = Field(default_factory=dict)
    last_interaction: Optional[datetime] = None


class ConversationContext(BaseModel):
    """Context for the current conversation."""

    user_profile: UserProfile
    recent_memories: list[Memory] = Field(default_factory=list)
    current_topic: Optional[str] = None
    conversation_history: list[str] = Field(default_factory=list)
