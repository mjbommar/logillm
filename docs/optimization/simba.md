# SIMBA: Stochastic Introspective Mini-Batch Ascent

SIMBA is LogiLLM's sophisticated hyperparameter and prompt optimization algorithm, adapted from DSPy but enhanced with zero-dependency Bayesian optimization and introspective rule generation.

## What SIMBA Does

SIMBA optimizes prompts through a multi-step process:

1. **Mini-batch sampling**: Tests configurations on small subsets of data
2. **Introspective rule generation**: Analyzes successful vs failed attempts 
3. **Demo appending**: Adds successful examples as few-shot demonstrations
4. **Multi-candidate generation**: Creates multiple variants per iteration
5. **Temperature-based selection**: Uses softmax sampling to pick promising candidates

## Key Features

- **Zero dependencies**: Pure Python implementation with no external packages
- **Parallel evaluation**: Concurrent processing of multiple candidates
- **Introspective learning**: LLM analyzes its own successes and failures
- **Adaptive sampling**: Smart selection of training examples
- **Trajectory tracking**: Full execution traces for debugging

## Basic Usage

```python
from logillm.optimizers import SIMBA
import asyncio

# Define success metric
def accuracy_metric(predicted, expected):
    return float(predicted.get("answer", "").strip().lower() == 
                expected.get("answer", "").strip().lower())

# Create SIMBA optimizer
optimizer = SIMBA(
    metric=accuracy_metric,
    bsize=16,              # Mini-batch size
    num_candidates=6,      # Candidates per iteration  
    max_steps=5,          # Number of optimization steps
    max_demos=4           # Max demonstrations per predictor
)

# Optimize module
result = await optimizer.optimize(
    module=qa_module,
    dataset=training_data,
    validation_set=test_data
)

print(f"Best score: {result.best_score:.3f}")
print(f"Improvement: {result.improvement:.3f}")
```

## Configuration Options

SIMBA supports extensive configuration through `SIMBAConfig`:

```python
from logillm.optimizers import SIMBA, SIMBAConfig

config = SIMBAConfig(
    bsize=32,                          # Mini-batch size
    num_candidates=8,                  # Candidates per iteration
    max_steps=10,                      # Optimization steps
    demo_input_field_maxlen=100_000,   # Max length for demo inputs
    num_threads=4,                     # Parallel evaluation threads
    temperature_for_sampling=0.3,      # Temperature for trajectory sampling
    temperature_for_candidates=0.2,    # Temperature for candidate selection
    
    # Base optimization config  
    max_demos=6,                       # Maximum demonstrations
    target_score=0.9,                  # Early stopping target
    early_stopping=True,               # Enable early stopping
    patience=3,                        # Patience for early stopping
)

optimizer = SIMBA(metric=accuracy_metric, config=config)
```

## The SIMBA Algorithm

### Step 1: Mini-Batch Generation
```python
# Sample random mini-batch from dataset
batch_size = 32
batch = random.sample(dataset, batch_size)
```

### Step 2: Program Trajectory Generation  
```python
# For each model variant and example, generate trajectory
for model in model_variants:
    for example in batch:
        # Select program probabilistically based on past performance
        program = softmax_sample(programs, scores, temperature=0.2)
        
        # Execute and record full trace
        result = await program(**example["inputs"])
        trajectory = get_execution_trace(program, example, result)
```

### Step 3: Performance-Based Bucketing
```python
# Group results by example, sort by performance
buckets = []
for example_idx in range(batch_size):
    example_results = [results[i] for i in range(example_idx, len(results), batch_size)]
    sorted_results = sorted(example_results, key=lambda r: r.score, reverse=True)
    
    # Calculate performance gaps
    max_score = sorted_results[0].score
    min_score = sorted_results[-1].score
    gap = max_score - min_score
    
    buckets.append((sorted_results, gap))

# Sort buckets by performance gaps (focus on high-variance examples)
buckets.sort(key=lambda b: b[1], reverse=True)
```

### Step 4: Strategy Application
SIMBA applies two main strategies to improve programs:

#### Demo Appending Strategy
Adds successful execution traces as demonstrations:

```python
async def append_demo(bucket, system):
    # Get best result from bucket
    best_result = bucket[0]
    trace = best_result.trace
    
    # Create demonstration from successful trace
    demo = Demonstration(
        inputs=best_result.inputs,
        outputs=best_result.outputs,
        score=best_result.score,
        metadata={"augmented": True}
    )
    
    # Add to predictor's demonstrations
    for predictor in system.predictors:
        predictor.demonstrations.append(demo)
```

#### Rule Appending Strategy  
Generates introspective rules by comparing good vs bad executions:

```python
async def append_rule(bucket, system):
    good, bad = bucket[0], bucket[-1]  # Best and worst
    
    # Compare execution traces
    better_trajectory = good.trace
    worse_trajectory = bad.trace
    
    # Generate advice using introspective module
    feedback = await generate_feedback(
        program_code=system.source,
        better_trajectory=better_trajectory,
        worse_trajectory=worse_trajectory,
        better_outputs=good.outputs,
        worse_outputs=bad.outputs
    )
    
    # Apply advice as instruction updates
    for module_name, advice in feedback.items():
        system.get_module(module_name).instruction += f"\n\n{advice}"
```

### Step 5: Candidate Evaluation
New candidate programs are evaluated on the same mini-batch:

```python
# Generate new program candidates using strategies
new_programs = []
for bucket in sorted_buckets:
    # Pick source program via softmax
    source = softmax_sample(existing_programs, scores, temperature=0.2)
    
    # Apply random strategy (demo appending or rule generation)
    strategy = random.choice([append_demo, append_rule])
    new_program = await strategy(bucket, source.copy())
    new_programs.append(new_program)

# Evaluate all new programs
scores = await evaluate_batch(new_programs, current_batch)
```

## Advanced Features

### Introspective Rule Generation

SIMBA's most sophisticated feature is automatic rule generation through introspection:

```python
# Example of generated introspective feedback
feedback = {
    "predictor_1": "When the question asks for reasoning, provide step-by-step logic before the final answer.",
    "predictor_2": "For mathematical problems, show your calculation explicitly.",
    "predictor_3": "When uncertain, acknowledge uncertainty rather than guessing."
}
```

This feedback is generated by analyzing execution traces where the model succeeded vs failed on similar inputs.

### Softmax Sampling for Exploration

SIMBA uses temperature-controlled softmax sampling to balance exploration and exploitation:

```python
def softmax_sample(programs, score_function, temperature=0.3):
    scores = [score_function(p) for p in programs]
    
    # Apply temperature and compute softmax
    exp_scores = [math.exp(s / temperature) for s in scores]
    probs = [e / sum(exp_scores) for e in exp_scores]
    
    # Sample based on probabilities
    return random.choices(programs, weights=probs)[0]
```

- **Low temperature** (0.1): Focuses on best programs (exploitation)
- **High temperature** (1.0): More random exploration
- **Default** (0.2-0.3): Good balance

### Parallel Evaluation

SIMBA evaluates multiple candidates concurrently:

```python
async def evaluate_batch(programs, examples):
    semaphore = asyncio.Semaphore(10)  # Limit concurrency
    
    async def eval_single(program, example):
        async with semaphore:
            try:
                result = await program(**example["inputs"])
                score = metric(result.outputs, example["outputs"])
                return {"program": program, "score": score, "result": result}
            except Exception as e:
                return {"program": program, "score": 0.0, "error": str(e)}
    
    tasks = [eval_single(p, ex) for p in programs for ex in examples]
    return await asyncio.gather(*tasks)
```

## Performance Tuning

### Mini-Batch Size (`bsize`)
- **Small (8-16)**: Faster iterations, less stable
- **Medium (32-64)**: Good balance
- **Large (128+)**: More stable, slower

### Number of Candidates (`num_candidates`)
- **Few (3-4)**: Fast but limited exploration
- **Medium (6-8)**: Default recommendation  
- **Many (10+)**: Thorough but expensive

### Temperature Settings
```python
config = SIMBAConfig(
    temperature_for_sampling=0.3,    # For selecting which programs to use
    temperature_for_candidates=0.2,  # For selecting source programs
)
```

Lower temperatures focus on proven performers, higher temperatures explore more.

## Comparison with DSPy SIMBA

| Feature | DSPy SIMBA | LogiLLM SIMBA |
|---------|------------|---------------|
| Dependencies | Requires many packages | Zero dependencies |
| Parallel evaluation | Basic | Full asyncio support |
| Introspective feedback | Basic | Enhanced with detailed analysis |
| Configuration | Limited | Extensive SIMBAConfig |  
| Error handling | Basic | Robust exception handling |
| Memory efficiency | Variable | Optimized for large datasets |

## Real-World Example

```python
import asyncio
from logillm import Predict
from logillm.optimizers import SIMBA

# Create a complex reasoning module
reasoning_module = Predict("""
question: str -> reasoning: str, confidence: float, answer: str
""")

# Dataset with reasoning problems
dataset = [
    {
        "inputs": {"question": "If a train travels 120 miles in 2 hours, what's its average speed?"},
        "outputs": {
            "reasoning": "Speed = distance / time = 120 miles / 2 hours = 60 mph",
            "confidence": 0.95,
            "answer": "60 mph"
        }
    },
    # ... more examples
]

# Success metric considering multiple outputs
def reasoning_metric(predicted, expected):
    # Check if answer is correct
    answer_correct = (predicted.get("answer", "").lower().strip() == 
                     expected.get("answer", "").lower().strip())
    
    # Check if reasoning mentions key concepts
    reasoning = predicted.get("reasoning", "").lower()
    expected_reasoning = expected.get("reasoning", "").lower()
    
    key_concepts = ["speed", "distance", "time", "miles", "hours"]
    reasoning_score = sum(1 for concept in key_concepts 
                         if concept in reasoning) / len(key_concepts)
    
    # Check confidence calibration
    confidence = predicted.get("confidence", 0.5)
    confidence_score = 1.0 - abs(confidence - (1.0 if answer_correct else 0.3))
    
    # Combined score
    return 0.5 * answer_correct + 0.3 * reasoning_score + 0.2 * confidence_score

# Configure SIMBA for complex reasoning
optimizer = SIMBA(
    metric=reasoning_metric,
    bsize=16,                         # Smaller batches for complex examples
    num_candidates=8,                 # More candidates for exploration
    max_steps=8,                      # More steps for refinement
    temperature_for_sampling=0.4,     # Higher temp for creative reasoning
    temperature_for_candidates=0.3,   # Medium temp for candidate selection
    max_demos=6                       # More demos for complex tasks
)

# Run optimization
async def optimize_reasoning():
    result = await optimizer.optimize(
        module=reasoning_module,
        dataset=dataset,
        validation_set=dataset[:20]
    )
    
    print(f"Optimization completed:")
    print(f"  Best score: {result.best_score:.3f}")
    print(f"  Improvement: {result.improvement:.3f}")
    print(f"  Optimization time: {result.optimization_time:.1f}s")
    
    # Show what SIMBA learned
    metadata = result.metadata
    print(f"\nOptimization details:")
    print(f"  Trials: {len(metadata['trial_logs'])}")
    print(f"  Final scores: {metadata['final_scores']}")
    
    return result.optimized_module

# Run optimization
optimized_reasoning = asyncio.run(optimize_reasoning())

# Test the optimized module
test_result = await optimized_reasoning(
    question="A car travels 300 kilometers in 4 hours. What is its speed in km/h?"
)

print(f"\nTest result:")
print(f"Reasoning: {test_result.outputs['reasoning']}")
print(f"Confidence: {test_result.outputs['confidence']}")
print(f"Answer: {test_result.outputs['answer']}")
```

## Next Steps

- [Hybrid Optimizer](hybrid-optimizer.md) - Combine SIMBA with hyperparameter optimization
- [Bootstrap Few-Shot](bootstrap-fewshot.md) - Alternative demonstration generation
- [COPRO](copro.md) - Collaborative prompt optimization
- [Search Strategies](strategies.md) - The Bayesian optimization algorithms SIMBA uses