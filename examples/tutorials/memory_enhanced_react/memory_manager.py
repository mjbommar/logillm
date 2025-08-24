"""Memory management system for persistent user memories and profiles."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ConversationContext, Memory, MemoryType, UserProfile


class MemoryManager:
    """Manages persistent user memories and profiles."""

    def __init__(self, storage_path: str = "memory_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.users: dict[str, UserProfile] = {}
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

    def store_memory(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType,
        tags: list[str] = None,
        importance: int = 1,
        metadata: dict = None,
    ) -> Memory:
        """Store a new memory for a user."""
        user = self.get_or_create_user(user_id)

        memory = Memory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            tags=tags or [],
            importance=importance,
            metadata=metadata or {},
        )

        user.memories.append(memory)
        user.last_interaction = datetime.now()
        self._save_user(user)

        return memory

    def search_memories(
        self,
        user_id: str,
        query: str = None,
        memory_type: MemoryType = None,
        tags: list[str] = None,
        limit: int = 10,
    ) -> list[Memory]:
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
        recent_memories = self.search_memories(user_id, query=query, limit=5)

        return ConversationContext(
            user_profile=user, recent_memories=recent_memories, current_topic=query
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
            importance=7,
        )

        self._save_user(user)
