"""Game models for AI text adventure."""

from enum import Enum

from pydantic import BaseModel, Field


class GameState(str, Enum):
    """Current state of the game."""

    EXPLORING = "exploring"
    COMBAT = "combat"
    DIALOGUE = "dialogue"
    INVENTORY = "inventory"
    GAME_OVER = "game_over"
    VICTORY = "victory"


class Player(BaseModel):
    """Player character information."""

    name: str = "Adventurer"
    health: int = Field(default=100, ge=0, le=100)
    inventory: list[str] = Field(default_factory=list)
    location: str = "Starting Village"
    experience: int = Field(default=0, ge=0)
    skills: dict[str, int] = Field(
        default_factory=lambda: {"strength": 10, "wisdom": 10, "charisma": 10}
    )


class GameContext(BaseModel):
    """Complete game context and state."""

    player: Player = Field(default_factory=Player)
    current_state: GameState = GameState.EXPLORING
    story_history: list[str] = Field(default_factory=list)
    available_actions: list[str] = Field(default_factory=list)
    current_scene: str = "You find yourself in a peaceful village..."
    turn_count: int = 0
