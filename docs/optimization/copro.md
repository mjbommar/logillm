# COPRO: Collaborative Prompt Optimization

COPRO (Collaborative Prompt Optimization) is LogiLLM's implementation of instruction optimization through breadth-first search and iterative refinement. It automatically generates and improves prompts by learning from previous attempts.

## What COPRO Does

COPRO optimizes the **instruction** component of prompts through a systematic process:

1. **Breadth-first generation**: Creates multiple instruction candidates simultaneously
2. **Evaluation and ranking**: Tests each candidate on the dataset
3. **Iterative refinement**: Uses previous attempts as feedback for better instructions
4. **Duplicate detection**: Removes redundant candidates
5. **Statistical tracking**: Monitors optimization progress and convergence

## Key Features

- **Breadth-first search**: Generates multiple candidates per iteration
- **Feedback-driven refinement**: Uses previous results to guide new generation
- **Temperature control**: Balances creativity and consistency
- **Duplicate removal**: Avoids wasting compute on repeated instructions
- **Statistics tracking**: Comprehensive optimization analytics

## Basic Usage

```python
from logillm.optimizers import COPRO
import asyncio

# Define success metric
def accuracy_metric(predicted, expected):
    return float(predicted.get("answer", "").strip().lower() == 
                expected.get("answer", "").strip().lower())

# Create COPRO optimizer
optimizer = COPRO(
    metric=accuracy_metric,
    breadth=8,              # Number of candidates per iteration
    depth=3,               # Number of refinement iterations
    init_temperature=1.2   # Temperature for instruction generation
)

# Optimize module instructions
result = await optimizer.optimize(
    module=qa_module,
    dataset=training_data,
    validation_set=test_data
)

print(f"Best instruction: {result.metadata['best_instruction']}")
print(f"Best score: {result.best_score:.3f}")
print(f"Improvement: {result.improvement:.3f}")
```

## The COPRO Algorithm

### Phase 1: Initial Breadth-First Generation

COPRO starts by generating multiple instruction variants:

```python
# Generate breadth-1 new instructions (e.g., 7 new + 1 original = 8 total)
candidates = []

# Use LLM to generate instruction variants
generation_prompt = f"""
Given this basic instruction: "{original_instruction}"

Generate {breadth-1} alternative instructions for the same task.
Each should be different in style, specificity, or approach.

Make them:
- Clear and actionable
- Appropriate for the task
- Different from each other
"""

# Generate and parse multiple instructions
generated = await instruction_generator(generation_prompt)
for instruction in generated.instructions:
    candidates.append(InstructionCandidate(
        instruction=instruction,
        prefix="",  # Can also generate output prefixes
        depth=0
    ))

# Add original as baseline
candidates.append(InstructionCandidate(
    instruction=original_instruction,
    prefix="",
    depth=0
))
```

### Phase 2: Evaluation and Scoring

Each candidate is evaluated on the dataset:

```python
evaluated_candidates = []

for candidate in candidates:
    # Create module variant with this instruction
    test_module = module.copy()
    set_instruction(test_module, candidate.instruction, candidate.prefix)
    
    # Evaluate on dataset
    total_score = 0
    for example in evaluation_set:
        prediction = await test_module(**example["inputs"])
        score = metric(prediction.outputs, example["outputs"])
        total_score += score
    
    candidate.score = total_score / len(evaluation_set)
    evaluated_candidates.append(candidate)
```

### Phase 3: Iterative Refinement

For each subsequent depth iteration, COPRO uses previous results as feedback:

```python
# Prepare attempt history for the LLM
attempts = []
best_candidates = sorted(evaluated_candidates, key=lambda c: c.score, reverse=True)

for i, candidate in enumerate(best_candidates[:breadth]):
    attempts.extend([
        f"Instruction #{i+1}: {candidate.instruction}",
        f"Prefix #{i+1}: {candidate.prefix}",
        f"Resulting Score #{i+1}: {candidate.score:.3f}"
    ])

# Generate refined instructions based on what worked/didn't work
refinement_prompt = f"""
Based on these previous attempts and their scores:

{chr(10).join(attempts)}

Generate {breadth} new improved instructions for the same task.
Learn from what worked well and avoid what performed poorly.

Focus on:
- Why did high-scoring instructions work better?
- What patterns emerge from successful attempts?
- How can you combine the best aspects?
"""

refined_candidates = await instruction_generator(refinement_prompt)
```

## Configuration Options

COPRO supports extensive configuration through `COPROConfig`:

```python
from logillm.optimizers import COPRO, COPROConfig

config = COPROConfig(
    breadth=10,                    # Candidates per iteration
    depth=4,                      # Refinement iterations  
    init_temperature=1.4,         # Generation temperature
    track_stats=True,             # Enable statistics tracking
    dedupe_candidates=True,       # Remove duplicates
    min_score_threshold=0.1,      # Minimum score to keep candidate
    
    # Base optimization config
    max_iterations=4,             # Same as depth
    target_score=0.9,            # Early stopping target
    early_stopping=True,         # Enable early stopping
    patience=2                   # Patience for early stopping
)

optimizer = COPRO(metric=accuracy_metric, config=config)
```

## Instruction Generation Strategies

COPRO uses two main generation strategies:

### 1. Basic Generation (Depth 0)
Simple instruction generation from scratch:

```python
class BasicGenerateInstruction(Signature):
    """Generate instruction variants for a task."""
    basic_instruction: str = InputField(description="Original instruction to improve")
    proposed_instruction: str = OutputField(description="New instruction variant")
    proposed_prefix: str = OutputField(description="Optional output prefix")
```

### 2. Attempt-Based Generation (Depth > 0)
Generation informed by previous attempts:

```python
class GenerateInstructionGivenAttempts(Signature):
    """Generate improved instructions based on previous attempts."""
    attempted_instructions: list[str] = InputField(
        description="Previous instructions and their scores"
    )
    proposed_instruction: str = OutputField(description="Improved instruction")
    proposed_prefix: str = OutputField(description="Optional output prefix")
```

## Advanced Features

### Duplicate Detection and Removal

COPRO automatically removes duplicate instructions:

```python
class InstructionCandidate:
    def __eq__(self, other):
        if not isinstance(other, InstructionCandidate):
            return False
        return (self.instruction == other.instruction and 
                self.prefix == other.prefix)
    
    def __hash__(self):
        return hash((self.instruction, self.prefix))

# Deduplication in action
def dedupe_candidates(candidates):
    seen = {}
    for candidate in candidates:
        key = (candidate.instruction, candidate.prefix)
        if key not in seen or candidate.score > seen[key].score:
            seen[key] = candidate
    return list(seen.values())
```

### Statistics Tracking

When enabled, COPRO tracks detailed optimization statistics:

```python
@dataclass
class COPROStats:
    results_best: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    results_latest: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    total_calls: int = 0

# Tracking in action
def track_stats(predictor_id, depth, latest_scores, all_scores):
    if not self.stats:
        return
        
    # Track latest iteration performance
    self.stats.results_latest[predictor_id]["depth"].append(depth)
    self.stats.results_latest[predictor_id]["max"].append(max(latest_scores))
    self.stats.results_latest[predictor_id]["average"].append(mean(latest_scores))
    self.stats.results_latest[predictor_id]["std"].append(stdev(latest_scores))
    
    # Track best candidates overall
    top_scores = sorted(all_scores, reverse=True)[:10]
    self.stats.results_best[predictor_id]["max"].append(max(top_scores))
    # ... more tracking
```

### Custom Prompt Models

You can use different LLMs for instruction generation:

```python
from logillm.providers import OpenAIProvider

# Use GPT-4 for instruction generation (higher quality)
prompt_provider = OpenAIProvider(model="gpt-4")

optimizer = COPRO(
    metric=accuracy_metric,
    breadth=6,
    depth=3,
    prompt_model=prompt_provider  # Custom model for generation
)
```

## Real-World Example

```python
import asyncio
from logillm import Predict
from logillm.optimizers import COPRO

# Create a classification module with basic instruction
classifier = Predict("""
text: str -> category: str, confidence: float
""")

# Set initial instruction
classifier.signature.instructions = "Classify the given text into the appropriate category."

# Dataset for sentiment analysis
dataset = [
    {
        "inputs": {"text": "I love this product! It's amazing."},
        "outputs": {"category": "positive", "confidence": 0.9}
    },
    {
        "inputs": {"text": "This is terrible. Waste of money."},
        "outputs": {"category": "negative", "confidence": 0.9}
    },
    {
        "inputs": {"text": "It's okay, nothing special."},
        "outputs": {"category": "neutral", "confidence": 0.7}
    },
    # ... more examples
]

# Multi-factor success metric
def classification_metric(predicted, expected):
    # Category accuracy (most important)
    category_correct = (predicted.get("category", "").lower() == 
                       expected.get("category", "").lower())
    
    # Confidence calibration (should be high when correct, lower when wrong)
    pred_confidence = predicted.get("confidence", 0.5)
    expected_confidence = expected.get("confidence", 0.5)
    confidence_error = abs(pred_confidence - expected_confidence)
    confidence_score = max(0, 1 - confidence_error)
    
    # Combined score (80% accuracy, 20% confidence)
    return 0.8 * category_correct + 0.2 * confidence_score

# Configure COPRO for classification
optimizer = COPRO(
    metric=classification_metric,
    breadth=8,               # Generate 8 instruction variants
    depth=4,                # 4 rounds of refinement
    init_temperature=1.3,   # Slightly higher creativity
    track_stats=True,       # Track detailed statistics
    dedupe_candidates=True  # Remove duplicates
)

# Run optimization
async def optimize_classifier():
    print("Starting COPRO optimization...")
    
    result = await optimizer.optimize(
        module=classifier,
        dataset=dataset,
        validation_set=dataset[:15]  # Use subset for validation
    )
    
    print(f"\nOptimization completed:")
    print(f"  Best instruction: {result.metadata['best_instruction']}")
    print(f"  Best score: {result.best_score:.3f}")
    print(f"  Improvement: {result.improvement:.3f}")
    print(f"  Total evaluations: {result.metadata['total_evaluations']}")
    
    # Show optimization progression
    print(f"\nOptimization progression:")
    scores = result.metadata['candidate_scores']
    print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"  Average score: {result.metadata['avg_score']:.3f}")
    print(f"  Score std dev: {result.metadata['std_score']:.3f}")
    
    return result.optimized_module

# Run optimization
optimized_classifier = asyncio.run(optimize_classifier())

# Test the optimized classifier
test_inputs = [
    "This product exceeded all my expectations!",
    "Complete garbage, don't buy this.",
    "It's decent but not worth the price."
]

print(f"\nTesting optimized classifier:")
for text in test_inputs:
    result = await optimized_classifier(text=text)
    print(f"  '{text}'")
    print(f"    Category: {result.outputs['category']}")
    print(f"    Confidence: {result.outputs['confidence']:.2f}")
```

## Performance Tips

### Breadth vs Depth Trade-offs

```python
# Fast exploration (good for simple tasks)
optimizer = COPRO(breadth=12, depth=2)  # 12 candidates, 2 iterations

# Deep refinement (good for complex tasks)  
optimizer = COPRO(breadth=6, depth=5)   # 6 candidates, 5 iterations

# Balanced approach (default recommendation)
optimizer = COPRO(breadth=8, depth=3)   # 8 candidates, 3 iterations
```

### Temperature Settings

```python
# Conservative (slight variations)
optimizer = COPRO(init_temperature=0.8)

# Balanced (default)
optimizer = COPRO(init_temperature=1.2)

# Creative (more radical changes)
optimizer = COPRO(init_temperature=1.6)
```

### Early Stopping

```python
config = COPROConfig(
    min_score_threshold=0.15,  # Stop generating candidates below 15%
    early_stopping=True,
    patience=2                 # Stop if no improvement for 2 iterations
)
```

## Comparison with DSPy

| Feature | DSPy COPRO | LogiLLM COPRO |
|---------|------------|---------------|
| Breadth-first search | ✅ | ✅ |
| Iterative refinement | ✅ | ✅ |
| Statistics tracking | Basic | Comprehensive |
| Duplicate detection | Basic | Hash-based deduplication |
| Custom prompt models | Limited | Full provider support |
| Configuration | Minimal | Extensive COPROConfig |
| Error handling | Basic | Robust exception handling |

LogiLLM's COPRO implementation provides more control, better error handling, and more detailed optimization insights.

## Next Steps

- [Hybrid Optimizer](hybrid-optimizer.md) - Combine COPRO with hyperparameter optimization
- [Bootstrap Few-Shot](bootstrap-fewshot.md) - Alternative demonstration-based optimization  
- [SIMBA](simba.md) - More sophisticated prompt evolution
- [Format Optimizer](format-optimizer.md) - Optimize prompt formats alongside instructions