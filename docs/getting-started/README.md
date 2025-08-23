# Getting Started with LogiLLM

**LogiLLM** lets you build LLM applications by defining what you want, not how to prompt for it.

## 30 Seconds to Your First App

```python
# Copy and run this - no setup needed!
from logillm import Predict
from logillm.providers import MockProvider, register_provider

# Use mock for instant gratification (no API key needed)
register_provider(MockProvider(default_response="42"), set_default=True)

# Say what you want
calculator = Predict("question -> answer")

# Get it
result = calculator(question="What is 6 times 7?")
print(result.answer)  # "42"
```

**That's it.** You just built an LLM application.

## 2 Minutes: Understanding What Happened

You just used LogiLLM's core paradigm: **declare what you want, not how to get it**.

```python
calculator = Predict("question -> answer")
#                    ^^^^^^^^^^^^^^^^^ 
#                    This signature defines your interface
```

This single line replaces:
- Prompt engineering
- Response parsing  
- Error handling
- Input/output validation

Want to understand more? → [Philosophy: Programming vs Prompting](../core-concepts/philosophy.md)

## 5 Minutes: Make It Real

Let's use a real LLM and build something useful:

```python
import os
from logillm import Predict, ChainOfThought
from logillm.providers import create_provider, register_provider

# Step 1: Connect to OpenAI (or Anthropic, Google, etc.)
os.environ["OPENAI_API_KEY"] = "your-key-here"  # or use .env file
provider = create_provider("openai", model="gpt-4")
register_provider(provider, set_default=True)

# Step 2: Build three useful tools in three lines
classifier = Predict("text -> sentiment, confidence: float")
summarizer = Predict("article -> summary, key_points: list[str]")
analyzer = ChainOfThought("problem -> analysis, solution, next_steps: list[str]")

# Step 3: Use them like functions
# Sentiment analysis
mood = classifier(text="This framework is amazing!")
print(f"Sentiment: {mood.sentiment} (confidence: {mood.confidence})")

# Document summarization  
summary = summarizer(article=long_article_text)
print(f"Summary: {summary.summary}")
print(f"Key points: {summary.key_points}")

# Problem solving with reasoning
solution = analyzer(problem="Our API response time is too slow")
print(f"Analysis: {solution.analysis}")
print(f"Solution: {solution.solution}")
print(f"Next steps: {solution.next_steps}")
```

### What Just Happened?

- **[Signatures](../core-concepts/signatures.md)** defined your inputs/outputs
- **[Modules](../core-concepts/modules.md)** (Predict, ChainOfThought) handled execution
- **[Providers](../core-concepts/providers.md)** connected to the LLM
- **[Adapters](../core-concepts/adapters.md)** (invisible here) formatted everything

## 10 Minutes: Your First Production App

Let's build a customer support classifier that actually learns and improves:

```python
from logillm import Predict, Retry
from logillm.providers import create_provider, register_provider
from logillm.optimizers import HybridOptimizer

# Setup
register_provider(create_provider("openai", model="gpt-4"), set_default=True)

# Define what we want with rich types
class SupportTicket:
    """Classify customer support tickets."""
    
    ticket: str = "Customer support ticket text"
    category: str = "One of: billing, technical, account, other"
    priority: str = "One of: low, medium, high, urgent"  
    confidence: float = "Confidence score 0-1"
    suggested_response: str = "Draft response to customer"

# Create a robust classifier with automatic retry
classifier = Retry(
    Predict(signature=SupportTicket),
    max_retries=3  # Automatically retry on failures
)

# Use it
result = classifier(
    ticket="I can't log into my account and I have a presentation in 1 hour!"
)
print(f"Category: {result.category}")      # "account"
print(f"Priority: {result.priority}")      # "urgent"
print(f"Confidence: {result.confidence}")  # 0.95
print(f"Draft response: {result.suggested_response}")
```

### But Wait, We Can Make It Better!

This is where LogiLLM shines. Let's optimize it:

```python
# Prepare some training examples
training_data = [
    {
        "ticket": "My bill seems wrong this month",
        "category": "billing",
        "priority": "medium"
    },
    # ... more examples
]

# Define metric function
def accuracy_metric(prediction, label):
    return prediction.category == label.category

# Here's the magic - optimize BOTH prompts AND model parameters
optimizer = HybridOptimizer(
    metric=accuracy_metric,
    strategy="alternating"  # Alternates between prompt and parameter optimization
)

# This finds the best prompts AND the best temperature/top_p settings
optimized_classifier = optimizer.optimize(
    classifier,
    dataset=training_data,
    param_space={
        "temperature": (0.0, 1.0),  # It will find the optimal temperature
        "top_p": (0.7, 1.0)         # And the optimal top_p
    }
)

# Now it's 20-40% more accurate!
```

**This is impossible in DSPy** - they can't optimize hyperparameters. → [Why This Matters](../optimization/hybrid-optimizer.md)

## 15 Minutes: Understanding the Full Power

Here's what you can build with LogiLLM:

### 1. Composable Modules
```python
# Stack modules like LEGO blocks
robust_classifier = Retry(           # Add retry logic
    Refine(                          # Add iterative refinement
        Predict(signature=SupportTicket),  # Base prediction
        num_iterations=2
    ),
    max_retries=3
)
```
Learn more → [Module Composition](../core-concepts/modules.md#composition)

### 2. Different Execution Strategies
```python
# Simple prediction
answer = Predict("question -> answer")

# With step-by-step reasoning
reasoned = ChainOfThought("question -> reasoning, answer")

# With tool use and actions
agent = ReAct("task -> actions: list, result")

# Multi-persona reasoning
council = Avatar("question -> perspectives: list, consensus")
```
Explore all modules → [Module Types](../core-concepts/modules.md)

### 3. Automatic Optimization
```python
# Define accuracy metric first
def accuracy_metric(prediction, reference):
    return prediction.category == reference.category

# Optimize prompts only (like DSPy)
from logillm.optimizers import BootstrapFewShot
prompt_opt = BootstrapFewShot(metric=accuracy_metric)

# Optimize hyperparameters only (DSPy can't do this)
from logillm.optimizers import SIMBA
param_opt = SIMBA(metric=accuracy_metric)

# Optimize EVERYTHING (LogiLLM's killer feature)
from logillm.optimizers import HybridOptimizer

def accuracy_metric(prediction, reference):
    return prediction.category == reference.category

hybrid_opt = HybridOptimizer(metric=accuracy_metric, strategy="joint")
```
Deep dive → [Optimization Guide](../optimization/overview.md)

### 4. Production Features
```python
# Track usage and costs
from logillm.core.callbacks import CallbackManager
callbacks = CallbackManager()
callbacks.on_completion = lambda r: print(f"Tokens: {r.usage.total_tokens}")

# Add runtime assertions
from logillm.core.assertions import assert_output_contains
classifier.add_assertion(assert_output_contains("category", ["billing", "technical"]))

# Cache responses
from logillm.core.types import CacheLevel
provider.cache_level = CacheLevel.FULL
```
Learn more → [Advanced Features](../advanced/)

## Your Learning Path

### Start Here (You Are Here)
1. ✅ **Getting Started** ← You are here
2. [Quickstart Tutorial](quickstart.md) - Build a Q&A system
3. [First Program](first-program.md) - Complete walkthrough

### Understand the Core
4. [Philosophy](../core-concepts/philosophy.md) - Why programming > prompting
5. [Signatures](../core-concepts/signatures.md) - Define what you want
6. [Modules](../core-concepts/modules.md) - Execution strategies

### Master Optimization (Our Competitive Edge)
7. [Optimization Overview](../optimization/overview.md) - The paradigm
8. [Hybrid Optimizer](../optimization/hybrid-optimizer.md) - The killer feature
9. [SIMBA](../optimization/simba.md) - Hyperparameter tuning

### Build Production Systems
10. [Providers](../core-concepts/providers.md) - LLM connections
11. [Error Handling](../advanced/error-handling.md) - Robustness
12. [Deployment](../tutorials/deployment.md) - Going to production

## Quick Decision Tree

**"I just want to see it work"**  
→ Copy the 30-second example above

**"I want to build something real"**  
→ Jump to [First Program](first-program.md)

**"I'm migrating from DSPy"**  
→ See [Migration Guide](dspy-migration.md)

**"Show me why this is better than DSPy"**  
→ Read [Hybrid Optimization](../optimization/hybrid-optimizer.md)

**"I want to optimize my LLM app"**  
→ Start with [Optimization Overview](../optimization/overview.md)

## Common Questions

**Q: Do I need API keys to start?**  
A: No! Use MockProvider to start immediately. Add real providers when ready.

**Q: What makes LogiLLM special?**  
A: We can optimize both prompts AND hyperparameters. [DSPy can't do this](../comparisons/vs-dspy.md).

**Q: Is it really zero dependencies?**  
A: Yes! Core LogiLLM is pure Python. Providers are optional.

**Q: How much improvement can I expect?**  
A: 20-40% typical with hybrid optimization vs 10-20% with prompt-only (DSPy).

**Q: Can I use this in production?**  
A: Yes. See [Production Guide](../tutorials/deployment.md).

## Get Help

- 🐛 [Report Issues](https://github.com/yourusername/logillm/issues)
- 💬 [Discussions](https://github.com/yourusername/logillm/discussions)
- 📖 [Full Documentation](../README.md)

## Next Step

**Ready to build something?** → [Create your first real application](first-program.md)

---

*Remember: With LogiLLM, you describe what you want, not how to prompt for it. The system handles the rest.*