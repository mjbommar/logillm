# Building an AI Text Adventure Game with LogiLLM

> **📍 Tutorial Path**: [Yahoo Finance ReAct](./yahoo-finance-react.md) → **AI Text Game** → [Memory-Enhanced ReAct](./memory-enhanced-react.md)  
> **⏱️ Time**: 20-25 minutes | **🎯 Difficulty**: Intermediate-Advanced  
> **💡 Concepts**: Interactive systems, Game state management, Dynamic narratives, User-driven experiences

This tutorial demonstrates how to create an interactive text-based adventure game using LogiLLM. We'll build a system that generates dynamic narratives, manages game state, and provides an engaging interactive experience.

**Perfect for**: Developers wanting to build interactive applications, game developers exploring AI, those ready to learn stateful systems.

**Builds on**: [Yahoo Finance ReAct](./yahoo-finance-react.md) - Now we'll take agent reasoning and apply it to dynamic, user-driven interactive storytelling.

## What You'll Build

By the end of this tutorial, you'll have a LogiLLM-powered text adventure game that can:

- **Generate dynamic storylines** based on player choices
- **Manage game state** including inventory, health, and progress
- **Create interactive dialogue** with NPCs and environments
- **Handle complex decision trees** with meaningful consequences
- **Save and load game sessions** using LogiLLM's persistence features
- **Adapt narratives** based on player history and preferences

## Prerequisites

- Python 3.9+ installed
- OpenAI or Anthropic API key
- Basic understanding of LogiLLM modules and signatures

## Installation and Setup

```bash
pip install logillm[openai]
# For rich console output (optional)
pip install rich
```

## Step 1: Game State and Models

```python
# models.py
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


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
    inventory: List[str] = Field(default_factory=list)
    location: str = "Starting Village"
    experience: int = Field(default=0, ge=0)
    skills: Dict[str, int] = Field(default_factory=lambda: {"strength": 10, "wisdom": 10, "charisma": 10})


class GameContext(BaseModel):
    """Complete game context and state."""
    player: Player = Field(default_factory=Player)
    current_state: GameState = GameState.EXPLORING
    story_history: List[str] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    current_scene: str = "You find yourself in a peaceful village..."
    turn_count: int = 0
```

## Step 2: LogiLLM Signatures for Game Logic

```python
# signatures.py
from logillm.core.signatures import Signature, InputField, OutputField
from typing import List, Dict, Any
from .models import GameState, Player


class GenerateScene(Signature):
    """Generate a new scene based on current game state."""
    
    player: Player = InputField(desc="Current player state")
    current_location: str = InputField(desc="Player's current location")
    story_history: List[str] = InputField(desc="Previous story events")
    player_action: str = InputField(desc="Action player wants to take")
    
    scene_description: str = OutputField(desc="Vivid description of the new scene")
    available_actions: List[str] = OutputField(desc="List of possible player actions")
    new_location: str = OutputField(desc="Updated location name")
    state_changes: Dict[str, Any] = OutputField(desc="Changes to player state (health, inventory, etc)")


class ProcessAction(Signature):
    """Process player action and determine outcomes."""
    
    player_action: str = InputField(desc="The action the player chose")
    current_context: Dict[str, Any] = InputField(desc="Current game context")
    scene_description: str = InputField(desc="Current scene")
    
    action_result: str = OutputField(desc="Description of what happens")
    success: bool = OutputField(desc="Whether the action succeeded")
    consequences: List[str] = OutputField(desc="List of consequences from the action")
    new_game_state: GameState = OutputField(desc="Updated game state")


class GenerateDialogue(Signature):
    """Generate NPC dialogue and conversation options."""
    
    npc_name: str = InputField(desc="Name of the NPC")
    player: Player = InputField(desc="Player information")
    conversation_context: str = InputField(desc="Context of the conversation")
    player_message: str = InputField(desc="What the player said")
    
    npc_response: str = OutputField(desc="NPC's response to the player")
    dialogue_options: List[str] = OutputField(desc="Player's response options")
    relationship_change: int = OutputField(desc="Change in NPC relationship (-5 to +5)")
```

## Step 3: Game Engine

```python
# game_engine.py
import asyncio
import json
from typing import Dict, List, Any, Optional
from logillm.core.predict import Predict
from logillm.core.modules import Module
from .signatures import GenerateScene, ProcessAction, GenerateDialogue
from .models import GameContext, GameState, Player


class TextAdventureEngine(Module):
    """Main game engine for the text adventure game."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Initialize LogiLLM components
        self.scene_generator = Predict(signature=GenerateScene)
        self.action_processor = Predict(signature=ProcessAction)
        self.dialogue_generator = Predict(signature=GenerateDialogue)
        
        # Game state
        self.context = GameContext()
        self.game_running = True
    
    def display_status(self) -> None:
        """Display current player status."""
        player = self.context.player
        print(f"\n{'='*50}")
        print(f"🧙 {player.name} | ❤️  {player.health}/100 | ⭐ Level {player.experience//100 + 1}")
        print(f"📍 Location: {player.location}")
        if player.inventory:
            print(f"🎒 Inventory: {', '.join(player.inventory)}")
        print(f"{'='*50}")
    
    def display_scene(self) -> None:
        """Display the current scene."""
        print(f"\n📖 {self.context.current_scene}")
        
        if self.context.available_actions:
            print(f"\n🎯 Available Actions:")
            for i, action in enumerate(self.context.available_actions, 1):
                print(f"   {i}. {action}")
    
    async def generate_new_scene(self, player_action: str) -> None:
        """Generate a new scene based on player action."""
        
        scene_result = await self.scene_generator(
            player=self.context.player,
            current_location=self.context.player.location,
            story_history=self.context.story_history,
            player_action=player_action
        )
        
        # Update game context
        self.context.current_scene = scene_result.scene_description
        self.context.available_actions = scene_result.available_actions
        self.context.player.location = scene_result.new_location
        
        # Apply state changes
        if scene_result.state_changes:
            for key, value in scene_result.state_changes.items():
                if key == "health" and isinstance(value, int):
                    self.context.player.health = max(0, min(100, self.context.player.health + value))
                elif key == "inventory_add" and isinstance(value, str):
                    self.context.player.inventory.append(value)
                elif key == "inventory_remove" and value in self.context.player.inventory:
                    self.context.player.inventory.remove(value)
                elif key == "experience" and isinstance(value, int):
                    self.context.player.experience += value
        
        # Add to story history
        self.context.story_history.append(f"Player: {player_action}")
        self.context.story_history.append(f"Result: {scene_result.scene_description}")
        
        # Keep history manageable
        if len(self.context.story_history) > 10:
            self.context.story_history = self.context.story_history[-8:]
    
    async def process_player_action(self, action: str) -> None:
        """Process the player's chosen action."""
        
        print(f"\n⚡ You choose: {action}")
        
        # Process the action
        action_result = await self.action_processor(
            player_action=action,
            current_context=self.context.dict(),
            scene_description=self.context.current_scene
        )
        
        print(f"\n📜 {action_result.action_result}")
        
        if action_result.consequences:
            print(f"\n⚠️  Consequences:")
            for consequence in action_result.consequences:
                print(f"   • {consequence}")
        
        # Update game state
        self.context.current_state = action_result.new_game_state
        
        # Check for game end conditions
        if self.context.player.health <= 0:
            self.context.current_state = GameState.GAME_OVER
            print(f"\n💀 GAME OVER! {self.context.player.name} has fallen...")
            self.game_running = False
        
        # Generate next scene
        await self.generate_new_scene(action)
        self.context.turn_count += 1
    
    async def start_game(self) -> None:
        """Start the main game loop."""
        
        print("🎮 Welcome to LogiLLM Text Adventure!")
        print("=" * 50)
        
        # Get player name
        player_name = input("Enter your character's name (or press Enter for 'Adventurer'): ").strip()
        if player_name:
            self.context.player.name = player_name
        
        print(f"\nWelcome, {self.context.player.name}! Your adventure begins...")
        
        # Generate initial scene
        await self.generate_new_scene("Look around and begin the adventure")
        
        # Main game loop
        while self.game_running and self.context.current_state not in [GameState.GAME_OVER, GameState.VICTORY]:
            
            self.display_status()
            self.display_scene()
            
            # Get player input
            try:
                if self.context.available_actions:
                    print(f"\nChoose an action (1-{len(self.context.available_actions)}) or type your own:")
                    user_input = input("> ").strip()
                    
                    # Parse input
                    if user_input.isdigit():
                        choice_num = int(user_input)
                        if 1 <= choice_num <= len(self.context.available_actions):
                            action = self.context.available_actions[choice_num - 1]
                        else:
                            print("Invalid choice! Please try again.")
                            continue
                    else:
                        action = user_input if user_input else "wait"
                else:
                    action = input("\nWhat do you want to do? > ").strip() or "wait"
                
                if action.lower() in ['quit', 'exit']:
                    print(f"Thanks for playing, {self.context.player.name}!")
                    break
                
                await self.process_player_action(action)
                
            except KeyboardInterrupt:
                print(f"\n\nThanks for playing, {self.context.player.name}!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                break
        
        print("\n🎭 Game ended. Thanks for playing!")
    
    def save_game(self, filename: str) -> None:
        """Save the current game state."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.context.dict(), f, indent=2)
            print(f"Game saved to {filename}")
        except Exception as e:
            print(f"Failed to save game: {e}")
    
    def load_game(self, filename: str) -> bool:
        """Load a saved game state."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.context = GameContext(**data)
            print(f"Game loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load game: {e}")
            return False
```

## Step 4: Demo Application

```python
# demo.py
import asyncio
import os
from logillm.providers import create_provider, register_provider
from .game_engine import TextAdventureEngine


async def demo_text_adventure() -> None:
    """Demonstrate the text adventure game."""
    
    # Setup LogiLLM provider
    model = os.environ.get("MODEL", "gpt-4o-mini")
    
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
    
    # Create and start game
    game = TextAdventureEngine()
    
    print("🎮 Starting LogiLLM Text Adventure Demo...")
    print("Note: This is an interactive game. Type 'quit' to exit.")
    print("=" * 60)
    
    await game.start_game()


async def main() -> None:
    """Main demo entry point."""
    await demo_text_adventure()


if __name__ == "__main__":
    asyncio.run(main())
```

## Key Features Demonstrated

This tutorial showcases:

1. **Interactive AI Storytelling**: Dynamic narrative generation based on player actions
2. **State Management**: Complex game state tracking and persistence
3. **Decision Trees**: Meaningful choices with consequences
4. **Modular Design**: Separate concerns for scene generation, action processing, and dialogue
5. **User Experience**: Clean console interface with rich formatting
6. **Persistence**: Save/load game functionality

## 🎓 What You've Learned

Outstanding! You've now mastered interactive AI system design:

✅ **Interactive Systems**: Built dynamic, user-driven applications  
✅ **State Management**: Created persistent, evolving game states  
✅ **Dynamic Narratives**: Generated adaptive content based on user choices  
✅ **User Experience**: Designed engaging interactive interfaces  
✅ **System Persistence**: Implemented save/load functionality for complex state

## 🚀 What's Next?

### Immediate Next Steps
**Ready for persistent memory?** → **[Memory-Enhanced ReAct Agent Tutorial](./memory-enhanced-react.md)**  
Learn how to take the interactive systems and state management you just mastered and add persistent memory that remembers across sessions.

### Apply What You've Learned
- **Create new game genres**: Build mystery games, sci-fi adventures, horror stories
- **Add multiplayer features**: Let multiple users interact with the same story
- **Expand game mechanics**: Add combat systems, puzzle-solving, character relationships

### Advanced Extensions
- **Visual interface**: Create web-based or GUI versions of the game
- **Voice integration**: Add speech-to-text and text-to-speech capabilities
- **AI-generated art**: Generate images for scenes and characters
- **Learning system**: Adapt storytelling based on player behavior patterns

### Tutorial Learning Path
1. ✅ **[LLM Text Generation](./llms-txt-generation.md)** - Foundation concepts
2. ✅ **[Email Extraction](./email-extraction.md)** - Structured data processing
3. ✅ **[Code Generation](./code-generation.md)** - Multi-step processing
4. ✅ **[Yahoo Finance ReAct](./yahoo-finance-react.md)** - Agent reasoning  
5. ✅ **AI Text Game** (You are here!)
6. → **[Memory-Enhanced ReAct Agent](./memory-enhanced-react.md)** - Persistent memory

### Concept Connections
- **From Interactive Games to Persistent Agents**: The state management you learned becomes user memory in conversational agents
- **Dynamic Content**: Narrative generation techniques apply to personalized AI assistants
- **User-Driven Systems**: Interactive patterns scale to chatbots and customer service agents

## 🛠️ Running the Tutorial

```bash
# With OpenAI
export OPENAI_API_KEY="your-key-here"
uv run --with logillm --with openai python -m examples.tutorials.ai_text_game.demo

# With Anthropic  
export ANTHROPIC_API_KEY="your-key-here"
uv run --with logillm --with anthropic python -m examples.tutorials.ai_text_game.demo

# Run tests to verify your setup
uv run --with logillm --with openai python examples/tutorials/ai_text_game/test_tutorial.py
```

---

**📚 [← Yahoo Finance ReAct](./yahoo-finance-react.md) | [Tutorial Index](./README.md) | [Memory-Enhanced ReAct Agent →](./memory-enhanced-react.md)**

You've mastered interactive AI systems! Ready to add persistent memory? Continue with **[Memory-Enhanced ReAct Agent](./memory-enhanced-react.md)** for the final advanced concept.