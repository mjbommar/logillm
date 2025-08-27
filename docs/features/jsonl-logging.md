# JSONL Logging

## Overview

LogiLLM provides comprehensive JSONL (JSON Lines) logging for tracking optimization processes, enabling reproducibility, debugging, and performance analysis. The JSONL format stores one JSON object per line, making it easy to process logs incrementally and analyze optimization runs.

## Key Features

- **Complete optimization tracking** - Records every step of the optimization process
- **Hyperparameter capture** - Logs all provider configuration settings
- **Instruction preservation** - Records signature docstrings and instructions
- **Prompt capture** - Can capture actual prompts sent to LLMs
- **Performance metrics** - Tracks scores, timing, and progression
- **Reproducibility** - Enables replay and analysis of optimization runs

## Basic Usage

```python
from logillm.core.jsonl_logger import OptimizationLogger
from logillm.core.predict import Predict
from logillm.optimizers.bootstrap_fewshot import BootstrapFewShot

# Create a module and optimizer
module = Predict("question -> answer")
optimizer = BootstrapFewShot(metric=accuracy_metric)

# Create logger
logger = OptimizationLogger(filepath="optimization.jsonl")

# Run optimization with logging
result = await logger.log_optimization(
    optimizer=optimizer,
    module=module,
    dataset=training_data,
    validation_set=validation_data
)
```

## JSONL Event Types

The logger records various event types throughout the optimization:

### `optimization_start`
Logged at the beginning of optimization, contains:
- Initial module configuration
- Signature and instructions
- Hyperparameters (temperature, max_tokens, etc.)
- Dataset size
- Timestamp

### `evaluation_start` / `evaluation_end`
Tracks each evaluation round:
- Iteration number
- Score achieved
- Number of traces evaluated
- Timing information

### `optimization_end`
Final event with complete results:
- Best score achieved
- Score improvement
- Final module state (with demonstrations)
- Total optimization time
- Score progression history

### `prompts_captured` (Optional)
When using `PromptCapturingLogger`:
- Actual prompts sent to LLM
- System messages and demonstrations
- Input formatting

## Advanced: Capturing Actual Prompts

For deeper insight into what's being sent to the LLM, use the `PromptCapturingLogger`:

```python
class PromptCapturingLogger(OptimizationLogger):
    """Extended logger that captures actual prompts."""
    
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.captured_prompts = []
    
    async def log_optimization(self, optimizer, module, dataset, **kwargs):
        # Wrap module's forward method to capture prompts
        original_forward = module.forward
        captured_prompts = self.captured_prompts
        
        async def capturing_forward(**inputs):
            # Enable debug mode to get prompts
            module._debug_mode = True
            result = await original_forward(**inputs)
            
            if result.prompt:
                captured_prompts.append({
                    "inputs": inputs,
                    "messages": result.prompt.get("messages", []),
                    "demos_count": result.prompt.get("demos_count", 0)
                })
            
            return result
        
        module.forward = capturing_forward
        
        try:
            result = await super().log_optimization(
                optimizer, module, dataset, **kwargs
            )
            
            # Log captured prompts
            if self.captured_prompts:
                self._write_event({
                    "event_type": "prompts_captured",
                    "total_prompts": len(self.captured_prompts),
                    "sample_prompts": self.captured_prompts[:3]
                })
            
            return result
        finally:
            module.forward = original_forward
```

## Analyzing JSONL Logs

### Reading Logs

```python
import json

# Read all events
with open("optimization.jsonl") as f:
    events = [json.loads(line) for line in f]

# Count event types
event_counts = {}
for event in events:
    event_type = event.get('event_type', 'unknown')
    event_counts[event_type] = event_counts.get(event_type, 0) + 1

# Extract scores
scores = [
    e['score'] for e in events 
    if e.get('event_type') == 'evaluation_end' and 'score' in e
]

# Get final configuration
final_event = events[-1]
if final_event['event_type'] == 'optimization_end':
    best_score = final_event['best_score']
    num_demos = final_event['final_module']['num_demos']
```

### Using Command-Line Tools

```bash
# View all event types
cat optimization.jsonl | jq '.event_type' | sort | uniq -c

# Extract scores
cat optimization.jsonl | jq 'select(.event_type=="evaluation_end") | .score'

# Pretty-print specific event
cat optimization.jsonl | jq 'select(.event_type=="optimization_end")'
```

## Configuration Tracking

The logger automatically captures:

### Hyperparameters
- `temperature` - Randomness in generation
- `max_tokens` - Maximum response length
- `top_p` - Nucleus sampling parameter
- `frequency_penalty` - Repetition penalty
- `presence_penalty` - Topic penalty

### Module Configuration
- Signature format (class-based or string)
- Instructions from docstrings
- Number of demonstrations
- Demo examples

## Best Practices

1. **Always use JSONL logging during development** - Helps debug optimization issues
2. **Store logs with descriptive names** - Include timestamp, module type, dataset
3. **Capture prompts for critical runs** - Use PromptCapturingLogger for detailed analysis
4. **Analyze progression** - Look for patterns in score improvements
5. **Version control logs** - Keep logs from important experiments

## Examples

See the complete examples:
- `examples/jsonl_logging_basic.py` - Simple usage demonstration
- `examples/jsonl_logging_comprehensive.py` - Advanced features including prompt capture

## Log Schema

```typescript
interface OptimizationStartEvent {
    event_type: "optimization_start"
    timestamp: string
    optimizer: string
    dataset_size: number
    initial_module: {
        type: string
        signature: string
        instructions: string | null
        hyperparameters: {
            temperature?: number
            max_tokens?: number
            top_p?: number
            // ...
        }
    }
}

interface EvaluationEndEvent {
    event_type: "evaluation_end"
    timestamp: string
    iteration: number
    score: number
    num_traces: number
}

interface OptimizationEndEvent {
    event_type: "optimization_end"
    timestamp: string
    success: boolean
    best_score: number
    improvement: number
    iterations: number
    optimization_time: number
    final_module: {
        num_demos: number
        demo_example?: {
            inputs: object
            outputs: object
        }
        hyperparameters: object
    }
}
```