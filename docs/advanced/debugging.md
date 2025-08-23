# Debugging LogiLLM Applications

*Tools and techniques for understanding what's happening inside your LLM programs*

## Quick Start: Debug Mode

The easiest way to debug LogiLLM applications is to enable debug mode, which captures the actual prompts sent to LLMs:

```python
from logillm.core.predict import Predict

# Method 1: Enable debug when creating a module
qa = Predict("question -> answer", debug=True)
result = await qa(question="What is 2+2?")
print(result.prompt)  # See the actual prompt sent to the LLM

# Method 2: Toggle debug mode dynamically
qa = Predict("question -> answer")
qa.enable_debug_mode()
result = await qa(question="What is 2+2?")
print(result.prompt["messages"])  # Access the messages

# Method 3: Use environment variable for global debug
# export LOGILLM_DEBUG=1
# Now all modules will have debug enabled by default
```

## Understanding Prompt Structure

When debug mode is enabled, the `prompt` field in Prediction contains:

```python
{
    "messages": [...],        # The actual messages sent to the LLM
    "adapter": "chat",        # Format adapter used (chat, json, xml, markdown)
    "demos_count": 2,         # Number of demonstrations included
    "provider": "openai",     # Provider being used
    "model": "gpt-4.1"       # Model being used
}
```

## Common Debugging Scenarios

### 1. Understanding Prompt Construction

When your module isn't producing expected results, check what's actually being sent:

```python
qa = Predict("question -> answer", debug=True)

# Add demonstrations to see how they're formatted
qa.add_demo({
    "inputs": {"question": "What is 2+2?"}, 
    "outputs": {"answer": "4"}
})

result = await qa(question="What is 5+3?")

# Examine the prompt
for i, msg in enumerate(result.prompt["messages"]):
    print(f"Message {i}: {msg['role']}")
    print(f"Content: {msg['content'][:200]}...")  # First 200 chars
```

### 2. Debugging Chain of Thought

See how reasoning is requested:

```python
from logillm.core.predict import ChainOfThought

cot = ChainOfThought("problem -> answer", debug=True)
result = await cot(problem="If I have 3 apples and buy 5 more, how many do I have?")

# The prompt will show the reasoning field was added
print("Reasoning field added:", "reasoning" in result.prompt["messages"][0]["content"])
```

### 3. Debugging Optimization

When optimizing modules, debug mode helps understand what changed:

```python
# Before optimization
original = Predict("text -> category", debug=True)
result1 = await original(text="This is a test")
original_prompt = result1.prompt["messages"][0]["content"]

# After optimization
from logillm.optimizers import BootstrapFewShot
optimizer = BootstrapFewShot()
optimized = await optimizer.optimize(original, dataset)

# Compare prompts
optimized.enable_debug_mode()
result2 = await optimized(text="This is a test")
optimized_prompt = result2.prompt["messages"][0]["content"]

print("Demos added:", result2.prompt["demos_count"] > 0)
print("Prompt changed:", original_prompt != optimized_prompt)
```

### 3a. Monitoring Optimization Progress

LogiLLM provides real-time optimization monitoring with **zero dependencies** (using only Python standard library):

```python
from logillm.optimizers import HybridOptimizer, AccuracyMetric

# Enable verbose mode to see step-by-step progress
optimizer = HybridOptimizer(
    metric=AccuracyMetric(key="category"),
    strategy="alternating",
    verbose=True  # Shows real-time optimization progress
)

# During optimization, you'll see:
# [   0.0s] Step   0/13 | Starting alternating optimization...
# [   0.1s] Step   0/13 | Baseline score: 0.3320
# [   0.2s] Step   1/13 | Iteration 1: Optimizing hyperparameters...
# [   2.1s] Step   1/10 | Testing params: temperature=0.723, top_p=0.850
# [   2.8s] Step   1/10 | 🎯 NEW BEST! Score: 0.7800
# [   3.5s] Step   2/10 | Testing params: temperature=0.451, top_p=0.920
# [   4.2s] Step   2/10 | Score: 0.7650
```

**Verbose mode is available for all optimizers:**
- `HybridOptimizer(verbose=True)` - Shows alternating optimization steps
- `HyperparameterOptimizer(verbose=True)` - Shows parameter trials
- `BootstrapFewShot(verbose=True)` - Shows demonstration generation
- `SIMBA(verbose=True)` - Shows evolution progress
- `COPRO(verbose=True)` - Shows instruction refinement

This logging uses **only Python's standard library** (no rich, tqdm, or other dependencies), maintaining LogiLLM's zero-dependency philosophy while providing essential visibility.

### 4. Debugging Different Adapters

See how different adapters format prompts:

```python
from logillm.core.adapters import AdapterFormat

# Chat adapter (default)
chat_qa = Predict("question -> answer", adapter="chat", debug=True)
result1 = await chat_qa(question="What is AI?")
print(f"Chat format: {result1.prompt['adapter']}")

# JSON adapter
json_qa = Predict("question -> answer", adapter="json", debug=True)
result2 = await json_qa(question="What is AI?")
print(f"JSON format: {result2.prompt['adapter']}")

# Compare the message formats
print("Chat message:", result1.prompt["messages"][0]["content"][:100])
print("JSON message:", result2.prompt["messages"][0]["content"][:100])
```

## Environment Variables

LogiLLM supports several environment variables for debugging:

```bash
# Enable debug mode globally
export LOGILLM_DEBUG=1

# Run your application
python your_app.py
```

With `LOGILLM_DEBUG=1`, all modules will have debug mode enabled by default unless explicitly disabled:

```python
# Even with LOGILLM_DEBUG=1, you can disable for specific modules
qa = Predict("question -> answer", debug=False)  # Overrides environment
```

## Performance Considerations

Debug mode has minimal performance impact:

- **Memory**: Prompts are only stored when debug is enabled
- **Speed**: No measurable impact on execution time
- **Security**: Be careful not to log prompts containing sensitive data

Best practices:
- Use debug mode during development and testing
- Disable in production unless troubleshooting
- Consider logging prompts to files instead of printing for large applications

## Advanced Debugging Techniques

### Custom Debug Handlers

Create a wrapper to process debug information:

```python
class DebugPredict(Predict):
    def __init__(self, *args, log_file="debug.log", **kwargs):
        super().__init__(*args, debug=True, **kwargs)
        self.log_file = log_file
    
    async def forward(self, **inputs):
        result = await super().forward(**inputs)
        
        # Log prompt to file
        if result.prompt:
            with open(self.log_file, "a") as f:
                import json
                f.write(json.dumps({
                    "timestamp": str(datetime.now()),
                    "inputs": inputs,
                    "prompt_size": len(str(result.prompt["messages"])),
                    "demos_count": result.prompt["demos_count"],
                    "success": result.success
                }) + "\n")
        
        return result
```

### Tracing + Debug Mode

Combine tracing with debug mode for complete visibility:

```python
qa = Predict("question -> answer", debug=True)
qa.enable_tracing()

result = await qa(question="What is the meaning of life?")

# Get both prompt and execution trace
print("Prompt:", result.prompt["messages"][0]["content"][:100])
print("Trace:", qa.get_trace())
```

## Troubleshooting Common Issues

### Issue: "Prompt is None even with debug=True"

Check that debug mode is actually enabled:
```python
print(f"Debug enabled: {qa.is_debugging()}")
```

### Issue: "Prompt too large to print"

Truncate or save to file:
```python
if result.prompt:
    messages = result.prompt["messages"]
    for msg in messages[:2]:  # First 2 messages only
        print(f"{msg['role']}: {msg['content'][:500]}...")  # First 500 chars
```

### Issue: "Want to see prompts for wrapped modules"

Enable debug on the inner module:
```python
base_qa = Predict("question -> answer", debug=True)
reliable_qa = Retry(base_qa, max_retries=3)

result = await reliable_qa(question="What is 2+2?")
# The prompt will be captured in the base module's prediction
```

## Summary

Debug mode in LogiLLM provides:
- **Visibility**: See exactly what's sent to LLMs
- **Flexibility**: Enable per-module, globally, or dynamically
- **Safety**: No performance impact when disabled
- **Integration**: Works with all modules and optimizers

Use debug mode during development to understand prompt construction, diagnose issues, and optimize your LLM applications effectively.