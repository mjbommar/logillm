# Callback System Documentation

## Overview

LogiLLM provides a comprehensive callback system that enables monitoring, logging, and customization of the execution flow across modules, optimizers, and providers. The callback system is designed to be flexible, performant, and easy to use.

## Key Features

- **Comprehensive Event Coverage**: Events for module execution, optimization, evaluation, and provider interactions
- **Priority-Based Execution**: Control callback execution order with priority levels
- **Context Propagation**: Automatic parent-child relationship tracking for nested operations
- **Thread-Safe**: Safe for use in concurrent environments
- **Minimal Overhead**: Less than 20% performance impact for lightweight callbacks
- **Async/Sync Support**: Works with both asynchronous and synchronous code

## Quick Start

### Basic Usage

```python
from logillm.core.callbacks import CallbackManager, AbstractCallback, ModuleEndEvent
from logillm.core.predict import Predict

# Define a custom callback
class SimpleLogger(AbstractCallback):
    async def on_module_end(self, event: ModuleEndEvent):
        print(f"Module {event.module.__class__.__name__} completed in {event.duration:.2f}s")

# Register the callback
manager = CallbackManager()
manager.register(SimpleLogger())

# Use modules as normal - callbacks fire automatically
module = Predict("question -> answer")
result = await module(question="What is AI?")
```

### JSONL Logging

For production use, LogiLLM provides a built-in JSONL logger:

```python
from logillm.core.jsonl_callback import JSONLCallback, register_jsonl_logger

# Quick setup with convenience function
callback_id = register_jsonl_logger(
    "execution.jsonl",
    include_module_events=True,
    include_provider_events=True
)

# Or use the class directly for more control
callback = JSONLCallback(
    filepath="detailed.jsonl",
    include_module_events=True,
    include_optimization_events=True,
    include_provider_events=False,
    append_mode=True
)
manager = CallbackManager()
manager.register(callback)
```

## Event Types

### Module Events

- **ModuleStartEvent**: Fired when a module begins execution
  - Contains: module, inputs, context
- **ModuleEndEvent**: Fired when a module completes
  - Contains: module, outputs, prediction, duration, success

### Optimization Events

- **OptimizationStartEvent**: Fired when optimization begins
  - Contains: optimizer, module, dataset, config
- **OptimizationEndEvent**: Fired when optimization completes
  - Contains: optimizer, result, duration, success
- **EvaluationStartEvent**: Fired when evaluation begins
  - Contains: optimizer, module, dataset
- **EvaluationEndEvent**: Fired when evaluation completes
  - Contains: optimizer, module, score, duration
- **HyperparameterUpdateEvent**: Fired when hyperparameters change
  - Contains: optimizer, module, parameters, iteration

### Provider Events

- **ProviderRequestEvent**: Fired before making an LLM request
  - Contains: provider, messages, parameters
- **ProviderResponseEvent**: Fired after receiving LLM response
  - Contains: provider, response, usage, duration
- **ProviderErrorEvent**: Fired when provider encounters an error
  - Contains: provider, error, messages

## Creating Custom Callbacks

### Basic Callback

```python
from logillm.core.callbacks import AbstractCallback

class CustomCallback(AbstractCallback):
    async def on_module_start(self, event):
        # Process module start
        pass
    
    async def on_module_end(self, event):
        # Process module end
        pass
    
    async def on_provider_request(self, event):
        # Monitor LLM requests
        pass
```

### Callback with Priority

```python
from logillm.core.callbacks import AbstractCallback, Priority

class HighPriorityCallback(AbstractCallback):
    @property
    def priority(self):
        return Priority.HIGH  # Executes before normal priority callbacks
    
    async def on_module_end(self, event):
        # Critical processing that should happen first
        pass
```

### Optimization Monitoring

```python
class OptimizationMonitor(AbstractCallback):
    def __init__(self):
        self.scores = []
        
    async def on_evaluation_end(self, event):
        self.scores.append(event.score)
        print(f"Evaluation score: {event.score:.3f}")
    
    async def on_optimization_end(self, event):
        if event.result:
            print(f"Optimization complete!")
            print(f"Best score: {event.result.best_score:.3f}")
            print(f"Improvement: {event.result.improvement:.1%}")
```

## Performance Considerations

### Overhead

The callback system is designed for minimal overhead:
- **Disabled callbacks**: Near-zero overhead (single boolean check)
- **Lightweight callbacks**: < 20% overhead
- **Heavy callbacks**: Overhead depends on callback implementation

### Best Practices

1. **Keep callbacks lightweight**: Avoid heavy computation in callbacks
2. **Use appropriate priority**: Lower priority callbacks run earlier
3. **Disable when not needed**: Use `CallbackManager().disable()` for performance-critical sections
4. **Batch operations**: Process data in batches rather than per-event

### Disabling Callbacks

```python
manager = CallbackManager()

# Temporarily disable callbacks
manager.disable()
# ... performance critical code ...
manager.enable()

# Or use environment variable
import os
os.environ["LOGILLM_CALLBACKS_ENABLED"] = "0"  # Disable globally
```

## Migration from OptimizationLogger

If you're using the legacy `OptimizationLogger` wrapper, migrate to callbacks:

### Old Approach (Wrapper)
```python
from logillm.core.jsonl_logger import OptimizationLogger

logger = OptimizationLogger("optimization.jsonl")
result = await logger.log_optimization(optimizer, module, dataset)
```

### New Approach (Callbacks)
```python
from logillm.core.jsonl_callback import OptimizationJSONLCallback
from logillm.core.callbacks import CallbackManager

# Register callback
callback = OptimizationJSONLCallback("optimization.jsonl")
CallbackManager().register(callback)

# Run optimization normally - callbacks fire automatically
result = await optimizer.optimize(module, dataset)
```

## Advanced Usage

### Context Propagation

The callback system automatically tracks context across nested operations:

```python
class ContextTracker(AbstractCallback):
    async def on_module_start(self, event):
        print(f"Call ID: {event.context.call_id}")
        if event.context.parent_call_id:
            print(f"Parent: {event.context.parent_call_id}")
```

### Multiple Callbacks

```python
manager = CallbackManager()

# Register multiple callbacks
manager.register(SimpleLogger())
manager.register(JSONLCallback("execution.jsonl"))
manager.register(OptimizationMonitor())

# All callbacks fire for each event
```

### Filtering Events

```python
from logillm.core.callbacks import CallbackType

# Register callback for specific event types only
manager.register(
    MyCallback(),
    callback_types=[CallbackType.MODULE_START, CallbackType.MODULE_END]
)
```

## Complete Example

```python
import asyncio
from logillm.core.callbacks import (
    CallbackManager, 
    AbstractCallback,
    Priority
)
from logillm.core.jsonl_callback import JSONLCallback
from logillm.core.predict import Predict
from logillm.providers.mock import MockProvider
from logillm.providers import register_provider

# Define custom callback
class ExecutionMonitor(AbstractCallback):
    def __init__(self):
        self.call_count = 0
        self.total_duration = 0
    
    async def on_module_end(self, event):
        self.call_count += 1
        self.total_duration += event.duration or 0
        
    def get_stats(self):
        return {
            "calls": self.call_count,
            "total_time": self.total_duration,
            "avg_time": self.total_duration / self.call_count if self.call_count > 0 else 0
        }

async def main():
    # Setup
    manager = CallbackManager()
    
    # Add JSONL logging
    manager.register(JSONLCallback("execution.jsonl"))
    
    # Add custom monitor
    monitor = ExecutionMonitor()
    manager.register(monitor)
    
    # Setup provider
    provider = MockProvider(response_text="AI is artificial intelligence")
    register_provider(provider, set_default=True)
    
    # Create and use module
    module = Predict("question -> answer")
    
    # Run multiple queries
    questions = [
        "What is AI?",
        "What is ML?",
        "What is DL?"
    ]
    
    for q in questions:
        result = await module(question=q)
        print(f"Q: {q}")
        print(f"A: {result.outputs['answer']}\n")
    
    # Get statistics
    stats = monitor.get_stats()
    print(f"Execution stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

### CallbackManager

Singleton manager for all callbacks:

- `register(callback, callback_types=None, priority=None)`: Register a callback
- `unregister(callback_id)`: Remove a callback
- `enable()`: Enable callback execution
- `disable()`: Disable callback execution
- `clear()`: Remove all callbacks

### AbstractCallback

Base class for callbacks with default no-op implementations:

- `on_module_start(event)`: Module execution started
- `on_module_end(event)`: Module execution completed
- `on_optimization_start(event)`: Optimization started
- `on_optimization_end(event)`: Optimization completed
- `on_evaluation_start(event)`: Evaluation started
- `on_evaluation_end(event)`: Evaluation completed
- `on_provider_request(event)`: Provider request made
- `on_provider_response(event)`: Provider response received
- `on_provider_error(event)`: Provider error occurred
- `on_hyperparameter_update(event)`: Hyperparameters updated

### Priority Levels

- `Priority.HIGHEST`: Executes first
- `Priority.HIGH`: Executes before normal
- `Priority.NORMAL`: Default priority
- `Priority.LOW`: Executes after normal
- `Priority.LOWEST`: Executes last

## Troubleshooting

### Callbacks Not Firing

1. Check callbacks are enabled:
```python
manager = CallbackManager()
print(manager.is_enabled())  # Should be True
```

2. Verify callback is registered:
```python
print(manager.get_registered_callbacks())
```

3. Check environment variable:
```python
import os
print(os.environ.get("LOGILLM_CALLBACKS_ENABLED", "1"))  # Should be "1"
```

### Performance Issues

If callbacks are causing performance problems:

1. Profile callback execution time
2. Reduce callback complexity
3. Use batching for expensive operations
4. Consider disabling non-critical callbacks
5. Use lower frequency sampling for metrics

### Memory Usage

For long-running processes:

1. Avoid storing large amounts of data in callbacks
2. Periodically flush data to disk
3. Use streaming/append mode for file-based callbacks
4. Clear callback state periodically

## Conclusion

The LogiLLM callback system provides powerful, flexible monitoring and customization capabilities while maintaining excellent performance. Whether you need simple logging, complex optimization monitoring, or custom integrations, the callback system has you covered.