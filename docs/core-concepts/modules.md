# LogiLLM Module System

*The building blocks of LLM programs*

## ELI5: What are Modules?

Think of modules like LEGO blocks for building AI programs:

- **Basic blocks** (like `Predict`) that do one thing really well
- **Complex blocks** (like `ChainOfThought`) that add reasoning capabilities  
- **Wrapper blocks** (like `Retry`) that make other blocks more reliable
- **Agent blocks** (like `ReAct`, `Avatar`) that can use tools and make decisions

Just like LEGO, you can combine these blocks to build sophisticated AI applications!

## Module Architecture

LogiLLM's module system provides a clean, composable architecture for building LLM programs. Every module:

- Takes inputs through a **signature** (like a function signature)
- Processes them through a **forward** method
- Returns structured **predictions** with outputs and metadata
- Can be **optimized** automatically to improve performance

```python
from logillm import Predict, ChainOfThought, Retry, ReAct

# Basic prediction module
qa = Predict("question -> answer")

# Chain of thought reasoning
cot_qa = ChainOfThought("question -> reasoning, answer")

# Retry wrapper for reliability
reliable_qa = Retry(qa, max_retries=3)

# ReAct agent with tools
agent = ReAct("task -> result", tools=[search_tool, calculator])
```

## Base Module Class

All modules inherit from the base `Module` class, which provides:

### Core Interface

```python
class Module:
    async def forward(self, **inputs) -> Prediction:
        """The main execution method - must be implemented by subclasses"""
        
    async def __call__(self, **inputs) -> Prediction:
        """User-facing call interface with validation and tracing"""
        
    def call_sync(self, **inputs) -> Prediction:
        """Synchronous wrapper for async forward method"""
```

### Key Features

**Async-First Design**: All modules are built for async execution with sync adapters:

```python
# Async usage (recommended)
result = await qa(question="What is the capital of France?")

# Sync usage (convenience wrapper)
result = qa.call_sync(question="What is the capital of France?")
```

**Automatic Tracing**: Track execution with built-in tracing:

```python
qa.enable_tracing()
result = await qa(question="Example question")
trace = qa.get_trace()  # Get detailed execution information
```

**Debug Mode**: Capture the actual prompts sent to LLMs (new in v0.2):

```python
# Enable debug when creating the module
qa = Predict("question -> answer", debug=True)
result = await qa(question="What is 2+2?")
print(result.prompt)  # See the actual prompt sent to the LLM

# Or toggle debug mode dynamically
qa.enable_debug_mode()
result = await qa(question="What is 5+3?")
print(result.prompt["messages"])  # Access the messages

qa.disable_debug_mode()
# Now prompts won't be captured

# Or use environment variable for global debug
# export LOGILLM_DEBUG=1
```

**Parameter Management**: Store learnable parameters for optimization:

```python
# Modules automatically track parameters that can be optimized
print(qa.optimization_parameters())  # Shows parameters available for tuning
```

## Core Module Types

### 1. Predict - The Foundation

`Predict` is the most fundamental module that orchestrates LLM calls:

```python
from logillm import Predict

# Simple string signature
qa = Predict("question -> answer")

# Class-based signature with types
from logillm.signatures import InputField, OutputField

class QASignature:
    question: str = InputField(desc="The question to answer")
    answer: str = OutputField(desc="The answer to the question")

qa = Predict(QASignature)
```

**What Predict does internally**:
1. **Input validation** using the signature
2. **Demo formatting** (few-shot examples)
3. **Provider calls** to the LLM
4. **Output parsing** from LLM response
5. **Trace capture** for monitoring

**Key features**:
- Works with any LLM provider (OpenAI, Anthropic, etc.)
- Supports different adapters (Chat, JSON, XML, Markdown)
- Automatic demo management for few-shot learning
- Built-in caching and retry logic
- Debug mode to capture actual prompts sent to LLMs

**Debug Mode Example**:
```python
# Enable debug to see what's actually sent to the LLM
qa = Predict("question -> answer", debug=True)
result = await qa(question="What is the capital of France?")

# Access the prompt information
print(f"Messages sent: {result.prompt['messages']}")
print(f"Adapter used: {result.prompt['adapter']}")
print(f"Number of demos: {result.prompt['demos_count']}")
```

### 2. ChainOfThought - Enhanced Reasoning

`ChainOfThought` extends `Predict` by adding a reasoning step:

```python
from logillm import ChainOfThought

# Automatically adds reasoning field
math_solver = ChainOfThought("problem -> answer")

# Custom reasoning field name
cot = ChainOfThought("question -> reasoning, answer", reasoning_field="reasoning")
```

**How it works**:
- Automatically modifies the signature to include a reasoning field
- Forces the LLM to show its work before giving the final answer
- Particularly effective for mathematical and logical problems

**Example usage**:
```python
result = await math_solver(problem="What is 15% of 240?")
print(result.reasoning)  # Shows step-by-step work
print(result.answer)     # Shows final answer
```

### 3. Retry - Error Recovery

`Retry` wraps any module with intelligent retry logic:

```python
from logillm import Retry, RetryStrategy

# Basic retry with exponential backoff
reliable_qa = Retry(qa, max_retries=3, strategy=RetryStrategy.EXPONENTIAL)

# Custom retry condition
def should_retry(prediction):
    return not prediction.success or "error" in prediction.outputs.get("answer", "")

custom_retry = Retry(qa, retry_condition=should_retry)
```

**Key features**:
- **Enhanced signatures**: Automatically adds `past_*` and `feedback` fields
- **Multiple strategies**: Immediate, linear, and exponential backoff
- **Error feedback**: Tells the LLM about previous failures
- **History tracking**: Monitors success rates and patterns

**Signature transformation**:
```python
# Original: question -> answer  
# Enhanced: question, past_answer, feedback -> answer
```

### 4. Refine - Iterative Improvement

`Refine` runs modules multiple times with different parameters to find the best output:

```python
from logillm import Refine

# Define a quality metric
def brevity_reward(inputs, prediction):
    if not prediction.success:
        return 0.0
    answer = prediction.outputs.get('answer', '')
    return max(0.0, 1.0 - len(answer.split()) * 0.1)  # Prefer shorter answers

# Create refined module
refined_qa = Refine(
    module=qa,
    N=3,  # Try 3 times
    reward_fn=brevity_reward,
    threshold=0.8  # Stop if we achieve this score
)
```

**Process**:
1. Runs module N times with different temperatures
2. Evaluates each attempt with the reward function
3. Returns first result above threshold OR best result
4. Generates feedback for future improvements

### 5. ReAct - Reasoning and Acting

`ReAct` implements the reasoning-and-acting pattern with tool use:

```python
from logillm import ReAct
from logillm.tools import Tool

# Define tools
def search(query: str) -> list[str]:
    """Search for information."""
    return [f"Result for {query}"]

def calculate(expression: str) -> float:
    """Evaluate math expressions."""
    return eval(expression)  # In real use, use safe evaluation

search_tool = Tool(search)
calc_tool = Tool(calculate)

# Create ReAct agent
agent = ReAct("task -> result", tools=[search_tool, calc_tool], max_iters=5)
```

**Execution flow**:
1. **Think**: Agent reasons about what to do next
2. **Act**: Agent chooses and executes a tool
3. **Observe**: Agent sees the tool's output
4. **Repeat**: Until task is complete or max iterations reached

**Example execution**:
```python
result = await agent(task="What is the population of Tokyo plus 15%?")
# Agent will:
# 1. Think: "I need to find Tokyo's population"
# 2. Act: search("Tokyo population")
# 3. Observe: "Tokyo has about 14 million people"
# 4. Think: "Now I need to calculate 15% more"
# 5. Act: calculate("14000000 * 1.15")
# 6. Observe: "16100000"
# 7. Think: "I have the answer"
# 8. Act: finish("Tokyo's population plus 15% is about 16.1 million")
```

### 6. Avatar - Tool-Using Agent

`Avatar` is a sophisticated agent that can use tools to accomplish complex tasks:

```python
from logillm import Avatar

# Create avatar with tools
assistant = Avatar(
    signature="task -> result",
    tools=[search_tool, calc_tool, file_tool],
    max_iters=5,
    verbose=True
)
```

**Key differences from ReAct**:
- **Dynamic signatures**: Adapts signature based on task history
- **Multi-step planning**: Better at complex, multi-step tasks
- **Tool management**: More sophisticated tool selection and error handling

## Advanced Features

### Module Composition

Modules can be easily composed to create complex pipelines:

```python
from logillm import ChainOfThought, Retry, Refine

# Create a reliable, refined reasoning system
base_qa = ChainOfThought("question -> reasoning, answer")
reliable_qa = Retry(base_qa, max_retries=2)
optimized_qa = Refine(reliable_qa, N=3, reward_fn=accuracy_reward)
```

### Batch Processing

All modules support batch processing for efficiency:

```python
questions = [
    {"question": "What is 2+2?"},
    {"question": "What is the capital of France?"},
    {"question": "Explain photosynthesis"}
]

results = await qa.batch_process(questions, batch_size=2)
```

### Configuration Management

Modules accept configuration for fine-tuning:

```python
qa = Predict(
    "question -> answer",
    config={
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9
    }
)
```

### Signature Resolution

LogiLLM handles multiple signature formats:

```python
# String format
qa1 = Predict("question -> answer")

# Class format
class QASignature:
    question: str = InputField()
    answer: str = OutputField()

qa2 = Predict(QASignature)

# Instance format  
qa3 = Predict(QASignature())
```

## Module Lifecycle

### States

Modules progress through several states:

```python
from logillm.types import ModuleState

print(qa.state)  # ModuleState.INITIALIZED

compiled_qa = qa.compile()
print(compiled_qa.state)  # ModuleState.COMPILED

optimized_qa = optimizer.optimize(qa, training_data)
print(optimized_qa.state)  # ModuleState.OPTIMIZED
```

### Validation

Modules provide comprehensive validation:

```python
# Check if module is properly configured
is_valid = qa.validate()
errors = qa.validation_errors()

if not is_valid:
    print(f"Module has issues: {errors}")
```

### Metrics and Monitoring

Built-in metrics tracking:

```python
qa.enable_tracing()

# Use the module
await qa(question="Test question")

# Get performance metrics
metrics = qa.get_metrics()
print(f"Success rate: {metrics['success_rate']}")
print(f"Average duration: {metrics['avg_duration']}s")
print(f"Total tokens used: {metrics['total_tokens']}")
```

## Tool Integration

### Creating Custom Tools

Tools are first-class citizens in LogiLLM:

```python
from logillm.tools import Tool

@Tool.decorator
def weather_tool(location: str) -> dict:
    """Get weather information for a location."""
    # In real implementation, call weather API
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "sunny"
    }

# Use with ReAct or Avatar
agent = ReAct("task -> result", tools=[weather_tool])
```

### Tool Registry

Manage collections of tools:

```python
from logillm.tools import ToolRegistry

registry = ToolRegistry()
registry.register(search_tool)
registry.register(calculator_tool)
registry.register(weather_tool)

# Use entire registry
agent = ReAct("task -> result", tools=registry)
```

## Comparison with DSPy

LogiLLM's module system improves on DSPy in several ways:

| Feature | DSPy | LogiLLM |
|---------|------|---------|
| **Async Support** | Limited | Native async-first |
| **Error Handling** | Basic | Comprehensive with retry/refine |
| **Tool Integration** | External | Built-in Tool class |
| **Signature Flexibility** | Rigid | Multiple formats supported |
| **Optimization** | Prompt-only | Hybrid (prompts + hyperparameters) |
| **Dependencies** | Heavy (Pydantic, others) | Zero-dependency core |
| **Module Composition** | Complex | Simple and intuitive |

## Performance Considerations

### Batch Processing

Use batch processing for multiple similar requests:

```python
# Efficient batch processing
results = await qa.batch_process(
    items=[{"question": q} for q in questions],
    batch_size=10  # Process 10 at a time
)
```

### Caching

Enable caching for repeated queries:

```python
from logillm.types import CacheLevel

qa = Predict(
    "question -> answer",
    config={"cache_level": CacheLevel.MEMORY}
)
```

### Provider Optimization

Choose providers wisely for your use case:

```python
# For simple tasks, use efficient models
simple_qa = Predict("question -> answer", provider=openai_3_5_turbo)

# For complex reasoning, use powerful models  
complex_qa = ChainOfThought("problem -> reasoning, answer", provider=gpt_4)
```

## Best Practices

### 1. Start Simple

Begin with basic modules and add complexity as needed:

```python
# Start with basic Predict
qa = Predict("question -> answer")

# Add reasoning if needed
qa = ChainOfThought("question -> reasoning, answer") 

# Add reliability if needed
qa = Retry(qa, max_retries=2)
```

### 2. Use Appropriate Signatures

Be specific in your signatures:

```python
# Vague (less effective)
general = Predict("input -> output")

# Specific (more effective)
qa = Predict("question: str -> answer: str")
math = Predict("problem: math problem -> solution: step by step solution")
```

### 3. Enable Tracing in Development

Always use tracing during development:

```python
qa.enable_tracing()
result = await qa(question="test")
trace = qa.get_trace()
# Analyze trace for optimization opportunities
```

### 4. Handle Errors Gracefully

Use retry and validation:

```python
qa = Retry(
    Predict("question -> answer"),
    max_retries=3,
    retry_condition=lambda p: p.success and len(p.outputs.get('answer', '')) > 10
)
```

### 5. Optimize Systematically

Use LogiLLM's optimization tools:

```python
from logillm.optimizers import HybridOptimizer

optimizer = HybridOptimizer(metric=accuracy_metric)
optimized_qa = optimizer.optimize(qa, training_data, validation_data)
```

The module system is the foundation of LogiLLM's power - by combining simple, well-designed modules, you can build sophisticated LLM applications that are reliable, optimizable, and maintainable.