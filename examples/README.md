# LogiLLM Examples

Learn LogiLLM through practical, working examples that demonstrate core concepts and advanced features.

## Quick Start

**Prerequisites:**
- OpenAI API key: `export OPENAI_API_KEY=your_key`
- Installation: `pip install logillm[openai]`

**Run any example:**
```bash
python examples/hello_world.py
```

## Example Categories

### 🚀 Getting Started
- **`hello_world.py`** - Your first LogiLLM program
  - Basic prediction modules
  - Structured outputs with types
  - Provider setup

### 📝 Core Concepts  
- **`signatures.py`** - Define what you want from your LLM
  - String signatures for quick prototyping
  - Type hints for structured data
  - Class-based signatures for complex outputs

- **`chain_of_thought.py`** - Step-by-step reasoning
  - Basic vs chain-of-thought comparison
  - Complex problem solving
  - Debug mode to see actual prompts

### 🎯 Advanced Features
- **`optimization.py`** - Make your models better automatically
  - Baseline performance testing
  - Hybrid optimization (prompts + hyperparameters)
  - Real-time progress monitoring

- **`few_shot.py`** - Learn from examples
  - Bootstrap learning
  - Performance comparison (before/after)
  - Automatic example selection

- **`persistence.py`** - Save and load your work
  - Save optimized modules to disk
  - Load modules for production use
  - Preserve few-shot examples and hyperparameters
  - **Essential for production workflows!**

- **`retry.py`** - Handle failures gracefully
  - Automatic retry strategies
  - Exponential backoff
  - Production error handling

## Learning Path

**New to LogiLLM?** Start here:
1. `hello_world.py` - Understand the basics
2. `signatures.py` - Learn to define structured outputs
3. `chain_of_thought.py` - See reasoning in action

**Ready for production?** Continue with:
4. `optimization.py` - Improve performance automatically
5. `few_shot.py` - Add learning capabilities
6. `persistence.py` - Save and load your optimized work
7. `retry.py` - Build robust applications

## API Keys

Examples use real LLM APIs to demonstrate actual functionality:

```bash
# Required for all examples
export OPENAI_API_KEY=your_openai_key

# Optional for future examples
export ANTHROPIC_API_KEY=your_anthropic_key
export GOOGLE_API_KEY=your_google_key
```

## Installation Options

```bash
# Just OpenAI (recommended for examples)
pip install logillm[openai]

# All providers
pip install logillm[all]

# Core only (no LLM providers)
pip install logillm
```

## Need Help?

Each example is heavily commented and self-contained. Run them to see LogiLLM in action, then modify them to explore different use cases.

**Example Output Preview:**
```
=== LogiLLM Hello World ===
Q: What is the capital of France?  
A: The capital of France is Paris.
Tokens: 66
```