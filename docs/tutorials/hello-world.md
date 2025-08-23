# Hello World: Your First LogiLLM Program

This tutorial gets you from zero to a working LogiLLM application in under 5 minutes.

## The Simplest Possible Example

```python
#!/usr/bin/env python3
# hello_world.py - Save this file and run with: python hello_world.py

from logillm import Predict
from logillm.providers import MockProvider, register_provider

# Use mock provider (no API key needed for testing)
mock = MockProvider(default_response="Hello, World!")
register_provider(mock, set_default=True)

# Define what you want: input -> output
greeter = Predict("name -> greeting")

# Use it
result = greeter(name="Alice")
print(result.greeting)  # "Hello, World!"
```

**That's it!** You have a working LLM application in 6 lines of code.

## Understanding Each Line

Let's break down what each line does:

```python
from logillm import Predict
```
**Source:** `logillm/core/predict.py:16-180`  
Imports the core prediction module that handles LLM interactions.

```python
from logillm.providers import MockProvider, register_provider
```
**Source:** `logillm/providers/mock.py:1-150`, `logillm/providers/registry.py:1-100`  
Imports the mock provider for testing and the registration system.

```python
mock = MockProvider(default_response="Hello, World!")
```
**Source:** `logillm/providers/mock.py:23-35`  
Creates a mock provider that returns a fixed response. Perfect for testing without API keys.

```python
register_provider(mock, set_default=True)
```
**Source:** `logillm/providers/registry.py:45-67`  
Registers the provider as the default for all modules to use.

```python
greeter = Predict("name -> greeting")
```
**Source:** `logillm/core/signatures/parser.py:100-200`  
Creates a module that takes a "name" input and produces a "greeting" output.

```python
result = greeter(name="Alice")
```
**Source:** `logillm/core/modules.py:100-143`  
Executes the module with the input and returns a Prediction object.

## Making It Real: Using OpenAI

To use a real LLM instead of the mock:

```python
#!/usr/bin/env python3
# hello_world_openai.py

import os
from logillm import Predict
from logillm.providers import create_provider, register_provider

# Set your API key (or use environment variable)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create and register OpenAI provider
provider = create_provider("openai", model="gpt-4")
register_provider(provider, set_default=True)

# Create a more interesting signature
qa = Predict("question -> answer")

# Ask a real question
result = qa(question="What is the meaning of life?")
print(f"Answer: {result.answer}")
```

## Step-by-Step Setup

### 1. Install LogiLLM

```bash
# Core installation (no dependencies!)
pip install logillm

# With OpenAI support
pip install logillm[openai]
```

### 2. Set Up Your Environment

```bash
# Create a .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Or export directly
export OPENAI_API_KEY="your-key-here"
```

### 3. Create Your First Program

Save this as `my_first_app.py`:

```python
from logillm import Predict, ChainOfThought
from logillm.providers import create_provider, register_provider

# Setup (runs once)
def setup():
    provider = create_provider("openai", model="gpt-4")
    register_provider(provider, set_default=True)

# Create modules
def create_modules():
    # Simple prediction
    classifier = Predict("text -> sentiment")
    
    # With reasoning
    analyzer = ChainOfThought("problem -> solution")
    
    return classifier, analyzer

# Main application
def main():
    setup()
    classifier, analyzer = create_modules()
    
    # Example 1: Sentiment analysis
    result = classifier(text="I love this new framework!")
    print(f"Sentiment: {result.sentiment}")
    
    # Example 2: Problem solving
    result = analyzer(problem="How do I learn Python?")
    print(f"Solution: {result.solution}")

if __name__ == "__main__":
    main()
```

### 4. Run It

```bash
python my_first_app.py
```

## Common Patterns

### Pattern 1: Question Answering
```python
qa = Predict("question -> answer")
result = qa(question="What is Python?")
```

### Pattern 2: Classification
```python
classifier = Predict("text -> category, confidence: float")
result = classifier(text="This movie was terrible")
print(f"{result.category} (confidence: {result.confidence})")
```

### Pattern 3: Extraction
```python
extractor = Predict("document -> summary, key_points: list")
result = extractor(document=long_text)
```

### Pattern 4: Multi-Step Reasoning
```python
reasoner = ChainOfThought("problem -> analysis, solution")
result = reasoner(problem="How do we reduce carbon emissions?")
```

## Troubleshooting

### No API Key Error
```python
# Problem: "No API key found"
# Solution: Set environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

### Import Error
```python
# Problem: "No module named 'openai'"
# Solution: Install with provider
pip install logillm[openai]
```

### Mock Provider Always Returns Same Thing
```python
# Problem: Mock returns fixed response
# Solution: That's intentional! Use for testing only
# For real responses, switch to OpenAI/Anthropic
```

## What's Next?

Now that you have a working example:

1. **Learn about Signatures** → [Core Concepts: Signatures](../core-concepts/signatures.md)
2. **Explore Modules** → [Core Concepts: Modules](../core-concepts/modules.md)
3. **Try Optimization** → [Optimization Overview](../optimization/overview.md)
4. **Build Something Real** → [First Program Tutorial](../getting-started/first-program.md)

## Quick Reference Card

```python
# Setup (once per app)
from logillm.providers import create_provider, register_provider
register_provider(create_provider("openai", model="gpt-4"))

# Basic prediction
from logillm import Predict
qa = Predict("question -> answer")

# With reasoning
from logillm import ChainOfThought  
cot = ChainOfThought("problem -> reasoning, solution")

# With retries
from logillm import Retry
robust = Retry(qa, max_retries=3)

# With refinement
from logillm import Refine
refined = Refine(qa, num_iterations=2)

# Optimization (advanced)
from logillm.optimizers import HybridOptimizer
optimizer = HybridOptimizer(metric=accuracy)
optimized = optimizer.optimize(qa, training_data)
```

## Summary

You've learned:
- ✅ How to create your first LogiLLM program
- ✅ The difference between mock and real providers  
- ✅ Basic signature patterns
- ✅ How to handle common issues
- ✅ Where each feature is implemented in the source code

**Total time to working app: 2 minutes**  
**Total code needed: 6 lines**  
**External dependencies: 0** (for mock provider)

Welcome to LogiLLM! 🚀