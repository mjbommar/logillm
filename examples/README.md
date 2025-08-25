# LogiLLM Examples

## Quick Start vs Deep Learning

Choose your path:

⏱️ **Execution Times**: Quick examples run in 2-15 seconds. Optimization examples may take 40-90+ seconds due to multiple API calls and iterative improvements.

### 🚀 Quick Examples (2-5 minutes)
Start here if you want to see LogiLLM work immediately:

- **`hello_world.py`** (17 lines) - Absolute minimum, matches [Step 1 of tutorial](../docs/getting-started/quickstart.md)
- **`basic_demo.py`** (36 lines) - Shows multiple features quickly
- **`few_shot_demo.py`** (50 lines) - Minimal few-shot learning
- **`minimal_math.py`** (58 lines) - Math solving with optimization
- **`signatures_demo.py`** (77 lines) - All signature definition patterns

### 📚 Detailed Tutorials (10-20 minutes)
Start here if you want to understand everything:

- **`few_shot.py`** - Comprehensive few-shot learning tutorial with explanations
- **`few_shot_math.py`** - Detailed math problem solving with performance metrics
- **`optimization.py`** - Multiple optimization strategies explained
- **`chain_of_thought.py`** - Deep dive into reasoning capabilities
- **`persistence.py`** - Production-ready save/load patterns

## Philosophy

Following DSPy's lead, we provide both:
- **Minimal examples** that demonstrate power in few lines (no hand-holding)
- **Detailed tutorials** for those who want to understand the "why"

## Complete Tutorial

For a comprehensive, step-by-step tutorial that builds from simple to complex:
**[docs/getting-started/quickstart.md](../docs/getting-started/quickstart.md)**

This 750+ line tutorial walks through building a complete production application.

## Learning Paths

### "Just Show Me" Path (5 minutes)
1. Run `hello_world.py` - See it work
2. Run `basic_demo.py` - See features  
3. Run `few_shot_demo.py` - See learning

### "I Want to Build" Path (30 minutes)
1. Read [quickstart.md](../docs/getting-started/quickstart.md) - Complete tutorial
2. Run examples that match tutorial steps
3. Modify examples for your use case

### "Deep Understanding" Path (2+ hours)
1. Start with detailed tutorials (`few_shot.py`, etc.)
2. Read the explanations and comments
3. Experiment with modifications
4. Check [API reference](../docs/api/)

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