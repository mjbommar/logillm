# Debugging LogiLLM Applications

*Tools and techniques for understanding what's happening inside your LLM programs*

## Quick Start: Debug Mode

The easiest way to debug LogiLLM applications is to enable debug mode, which captures **complete request and response data** from LLM API interactions:

```python
from logillm.core.predict import Predict

# Method 1: Enable debug when creating a module
qa = Predict("question -> answer", debug=True)
result = await qa(question="What is 2+2?")

# Access complete debug information
print("Request:", result.request)    # Full request payload sent to API
print("Response:", result.response)  # Complete response received from API
print("Prompt:", result.prompt)      # Formatted prompt information

# Method 2: Toggle debug mode dynamically
qa = Predict("question -> answer")
qa.enable_debug_mode()
result = await qa(question="What is 2+2?")
print("Tokens used:", result.response["usage"]["total_tokens"])

# Method 3: Use environment variable for global debug
# export LOGILLM_DEBUG=1
# Now all modules will have debug enabled by default
```

## Understanding Debug Data Structure

When debug mode is enabled, LogiLLM captures **complete request and response data** from LLM API interactions:

### Request Data (`result.request`)
Contains the complete payload sent to the LLM provider:

```python
{
    "messages": [...],           # Full conversation messages
    "provider": "openai",        # Provider identifier
    "model": "gpt-4.1-mini",     # Model being used
    "adapter": "chat",           # Message format adapter
    "demos_count": 0,            # Number of demonstrations included
    "provider_config": {...},    # All provider parameters (temperature, etc.)
    "timestamp": "..."           # When request was made
}
```

### Response Data (`result.response`)
Contains the complete response received from the LLM provider:

```python
{
    "text": "...",               # Full response text
    "usage": {                   # Complete token usage breakdown
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 15
    },
    "cost": 0.00023,            # API cost in dollars (if available)
    "latency": 0.85,            # Response time in seconds
    "finish_reason": "stop",    # Why generation stopped
    "model": "gpt-4.1-mini",    # Responding model
    "provider": "openai",       # Responding provider
    "metadata": {...},          # Provider-specific metadata
    "timestamp": "..."          # When response was received
}
```

### Prompt Data (`result.prompt`) - Legacy
For backward compatibility, the original prompt structure is still available:

```python
{
    "messages": [...],        # The actual messages sent to the LLM
    "adapter": "chat",        # Format adapter used
    "demos_count": 2,         # Number of demonstrations included
    "provider": "openai",     # Provider being used
    "model": "gpt-4.1"       # Model being used
}
```

## Common Debugging Scenarios

### 1. Understanding Complete API Interactions

When debug mode is enabled, you get **complete visibility** into LLM API interactions:

```python
qa = Predict("question -> answer", debug=True)
result = await qa(question="What is the capital of France?")

# Complete request data
print("📤 REQUEST:")
print(f"  Provider: {result.request['provider']}")
print(f"  Model: {result.request['model']}")
print(f"  Messages: {len(result.request['messages'])}")
print(f"  Timestamp: {result.request['timestamp']}")

# Complete response data
print("\n📥 RESPONSE:")
print(f"  Text: {result.response['text'][:100]}...")
print(f"  Tokens: {result.response['usage']['total_tokens']}")
print(f"  Cost: ${result.response.get('cost', 'N/A')}")
print(f"  Latency: {result.response.get('latency', 'N/A')}s")
print(f"  Finish Reason: {result.response['finish_reason']}")
```

### 2. Monitoring API Costs and Performance

Track costs and performance across your application:

```python
qa = Predict("question -> answer", debug=True)

# Make several predictions
results = []
for question in ["What is 2+2?", "What is 3+3?", "What is 4+4?"]:
    result = await qa(question=question)
    results.append(result)

# Analyze costs and performance
total_cost = sum(r.response.get('cost', 0) for r in results if r.response)
total_tokens = sum(r.response['usage']['total_tokens'] for r in results if r.response)
avg_latency = sum(r.response.get('latency', 0) for r in results if r.response) / len(results)

print(f"Total cost: ${total_cost:.6f}")
print(f"Total tokens: {total_tokens}")
print(f"Average latency: {avg_latency:.2f}s")
```

### 3. Debugging API Errors and Issues

Get complete information when API calls fail:

```python
qa = Predict("question -> answer", debug=True)

try:
    result = await qa(question="Very long question that might exceed token limits...")
except Exception as e:
    # Even on errors, debug data is captured
    if hasattr(result, 'request') and result.request:
        print("Request that caused error:")
        print(f"  Message length: {len(str(result.request['messages']))} chars")
        print(f"  Provider: {result.request['provider']}")
    print(f"Error: {e}")
```

### 4. Understanding Prompt Construction

When your module isn't producing expected results, check what's actually being sent:

```python
qa = Predict("question -> answer", debug=True)

# Add demonstrations to see how they're formatted
qa.add_demo({
    "inputs": {"question": "What is 2+2?"},
    "outputs": {"answer": "4"}
})

result = await qa(question="What is 5+3?")

# Examine the complete request
for i, msg in enumerate(result.request["messages"]):
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

Debug mode in LogiLLM provides **complete transparency** into LLM API interactions:

### 🔍 **Complete Request Logging**
- Full messages sent to LLM APIs
- Provider and model information
- All request parameters and configurations
- Timestamps for request tracking

### 📥 **Complete Response Logging**
- Full response text from LLMs
- Detailed token usage (input, output, cached, reasoning)
- API costs and latency metrics
- Finish reasons and metadata
- Response timestamps

### 🎛️ **Flexible Control**
- **Per-module**: `debug=True` parameter
- **Global**: `LOGILLM_DEBUG=1` environment variable
- **Dynamic**: `enable_debug_mode()` / `disable_debug_mode()` methods
- **Zero performance impact** when disabled

### 🔗 **Universal Support**
- Works with **all LLM providers** (OpenAI, Anthropic, Google, etc.)
- Compatible with **all module types** (Predict, Avatar, ReAct, Retry, etc.)
- Integrates with **all optimizers** and workflows

### 💡 **Use Cases**
- **Development**: Understand prompt construction and API behavior
- **Debugging**: Diagnose issues with LLM responses and parameters
- **Monitoring**: Track costs, performance, and usage patterns
- **Optimization**: Analyze token usage and API efficiency
- **Compliance**: Audit LLM interactions and data flows

Debug mode gives you **complete visibility** into your LLM applications, making development, debugging, and optimization much more effective!