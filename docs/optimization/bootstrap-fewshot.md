# Bootstrap Few-Shot: Teacher-Student Demonstration Generation

Bootstrap Few-Shot is LogiLLM's implementation of DSPy's most important optimizer. It uses a teacher-student paradigm to automatically generate high-quality few-shot demonstrations by bootstrapping from the model's own successful attempts.

## The Core Insight

LLMs can often solve problems correctly *sometimes*, even without examples. Bootstrap Few-Shot exploits this by:

1. **Teacher phase**: Run the model many times with higher temperature (creativity)
2. **Success filtering**: Keep only the attempts that succeed according to your metric
3. **Student phase**: Use successful attempts as few-shot demonstrations for the final model

This transforms the model's occasional successes into consistent performance.

## Key Features

- **Temperature scheduling**: Adaptive temperature decay for better exploration
- **Diversity-aware selection**: Balances accuracy and demonstration variety
- **Rescue mode**: Special handling for challenging baselines
- **Teacher-student architecture**: Separates exploration from exploitation
- **Zero dependencies**: Pure Python implementation

## Basic Usage

```python
from logillm.optimizers import BootstrapFewShot
import asyncio

# Define success metric
def accuracy_metric(predicted, expected):
    return float(predicted.get("answer", "").strip().lower() == 
                expected.get("answer", "").strip().lower())

# Create bootstrap optimizer
optimizer = BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=4,  # Number of demos to generate
    max_labeled_demos=8,       # Max labeled demos to supplement with
    teacher_settings={"temperature": 1.2}  # Higher temp for exploration
)

# Bootstrap demonstrations from successful attempts
result = await optimizer.optimize(
    module=qa_module,
    dataset=training_data,
    validation_set=test_data
)

print(f"Generated {result.metadata['num_demos']} demonstrations")
print(f"Improvement: {result.improvement:.2%}")
```

## The Bootstrap Algorithm

### Step 1: Teacher Configuration

The teacher model uses higher temperature for creative exploration:

```python
# Create teacher (copy of student with different settings)
teacher = module.deepcopy()

# Apply teacher settings
teacher_settings = {
    "temperature": 1.2,      # Higher creativity
    "top_p": 0.9,           # Allow diverse tokens
    "max_tokens": 200        # Sufficient response length
}

# Update teacher configuration
teacher.config.update(teacher_settings)
```

### Step 2: Bootstrap Generation with Temperature Scheduling

LogiLLM enhances basic bootstrapping with adaptive temperature:

```python
# Temperature scheduling parameters
initial_temperature = 1.2    # Start creative
temperature_decay = 0.9      # Decay factor per round
min_temperature = 0.3        # Minimum temperature
rescue_mode = False          # Special mode for poor baselines

demonstrations = []
current_temperature = initial_temperature

for round_idx in range(max_rounds):
    print(f"Round {round_idx + 1}: temperature = {current_temperature:.2f}")
    
    # Update teacher temperature for this round
    teacher.provider.temperature = current_temperature
    
    # Try to generate demonstrations
    attempts = 0
    successful_demos = 0
    
    while (successful_demos < target_demos and 
           attempts < max_attempts):
        
        # Pick random example
        example = random.choice(dataset)
        
        # Run teacher
        try:
            prediction = await teacher(**example["inputs"])
            
            if prediction.success:
                # Evaluate success
                score = metric(prediction.outputs, example["outputs"])
                
                if score >= success_threshold:
                    # Create demonstration
                    demo = Demonstration(
                        inputs=example["inputs"],
                        outputs=prediction.outputs,
                        score=score,
                        metadata={
                            "teacher": True,
                            "temperature": current_temperature,
                            "attempt": attempts
                        }
                    )
                    demonstrations.append(demo)
                    successful_demos += 1
                    
        except Exception:
            pass  # Continue on errors
            
        attempts += 1
    
    # Apply temperature decay
    current_temperature = max(min_temperature, 
                            current_temperature * temperature_decay)
    
    # Stop if we have enough demonstrations
    if len(demonstrations) >= max_bootstrapped_demos:
        break
```

### Step 3: Diversity-Aware Selection

LogiLLM goes beyond simple score-based selection by considering diversity:

```python
def select_demonstrations_with_diversity(demos, max_demos, diversity_weight=0.3):
    """Select demonstrations balancing accuracy and diversity."""
    selected = []
    remaining = list(demos)
    
    for i in range(min(max_demos, len(demos))):
        if not remaining:
            break
            
        # Score each remaining demo
        scores = []
        for demo in remaining:
            # Base score (accuracy)
            accuracy_score = demo.score
            
            # Diversity score (dissimilarity to selected demos)
            diversity_score = 0.0
            if selected:
                similarities = []
                for selected_demo in selected:
                    # Compare inputs and outputs
                    input_sim = text_similarity(
                        str(demo.inputs), str(selected_demo.inputs)
                    )
                    output_sim = text_similarity(
                        str(demo.outputs), str(selected_demo.outputs)
                    )
                    similarities.append((input_sim + output_sim) / 2)
                
                # Diversity is inverse of max similarity
                diversity_score = 1.0 - max(similarities)
            else:
                diversity_score = 1.0  # First demo gets full diversity score
            
            # Combined score
            combined = ((1 - diversity_weight) * accuracy_score + 
                       diversity_weight * diversity_score)
            scores.append((combined, demo))
        
        # Select best and remove from remaining
        scores.sort(key=lambda x: x[0], reverse=True)
        best_demo = scores[0][1]
        selected.append(best_demo)
        remaining.remove(best_demo)
    
    return selected
```

## Advanced Configuration

Bootstrap Few-Shot supports extensive configuration through `BootstrapFewShotConfig`:

```python
from logillm.optimizers import BootstrapFewShot, BootstrapFewShotConfig

config = BootstrapFewShotConfig(
    # Core bootstrap parameters
    max_bootstrapped_demos=6,        # Demos to generate
    max_labeled_demos=10,            # Labeled demos to add if needed
    max_rounds=2,                    # Bootstrap rounds
    metric_threshold=0.7,            # Success threshold for demos
    
    # Temperature scheduling
    initial_teacher_temperature=1.0,  # Starting temperature
    temperature_decay=0.9,           # Decay per round
    min_temperature=0.3,             # Minimum temperature
    
    # Rescue mode (for challenging baselines)
    rescue_mode_threshold=0.2,       # Trigger rescue if baseline < 20%
    rescue_initial_temperature=1.5,  # Higher temp for rescue mode
    rescue_max_attempts_multiplier=2.0,  # More attempts in rescue mode
    
    # Diversity scoring
    use_diversity_scoring=True,      # Enable diversity-aware selection
    diversity_weight=0.3,            # Weight for diversity vs accuracy
    
    # Base optimization config
    early_stopping=True,
    patience=2,
    target_score=0.85
)

optimizer = BootstrapFewShot(metric=accuracy_metric, config=config)
```

## Rescue Mode for Challenging Tasks

When the baseline model performs very poorly (< 20% success), Bootstrap Few-Shot activates "rescue mode":

```python
# Check if rescue mode is needed
baseline_score = await evaluate(module, validation_set)
rescue_mode = baseline_score < 0.2

if rescue_mode:
    print(f"Baseline score {baseline_score:.1%} - activating rescue mode!")
    
    # Rescue mode settings
    teacher_temperature = 1.5      # Even higher creativity
    max_attempts = max_attempts * 2  # Try much harder
    
    # May also use different success threshold
    success_threshold = 0.4  # Lower bar for "success"
```

Rescue mode helps with tasks where the model initially struggles to succeed even occasionally.

## Demonstration Quality Analysis

LogiLLM provides detailed analysis of generated demonstrations:

```python
# After optimization, analyze demonstration quality
metadata = result.metadata

print(f"Demonstration Analysis:")
print(f"  Total generated: {metadata['total_attempts']}")  
print(f"  Successfully used: {metadata['num_demos']}")
print(f"  Success rate: {metadata['num_demos'] / metadata['total_attempts']:.1%}")

print(f"  Score distribution:")
demo_scores = metadata['demo_scores']
print(f"    Min: {min(demo_scores):.3f}")
print(f"    Max: {max(demo_scores):.3f}")
print(f"    Avg: {metadata['avg_demo_score']:.3f}")

print(f"  Source breakdown:")
print(f"    Bootstrapped: {metadata['num_bootstrapped']}")
print(f"    Labeled: {metadata['num_labeled']}")

print(f"  Temperature schedule:")
temp_schedule = metadata['temperature_schedule']
print(f"    Initial: {temp_schedule['initial']:.2f}")
print(f"    Final: {temp_schedule['final']:.2f}")
print(f"    Decay: {temp_schedule['decay']:.2f}")
```

## Real-World Example

```python
import asyncio
from logillm import Predict
from logillm.optimizers import BootstrapFewShot

# Create a math reasoning module
math_solver = Predict("""
problem: str -> reasoning: str, answer: str
""")

# Dataset with math word problems
dataset = [
    {
        "inputs": {"problem": "Sarah has 12 apples. She gives 3 to her friend and eats 2. How many does she have left?"},
        "outputs": {
            "reasoning": "Sarah started with 12 apples. She gave away 3 and ate 2, so she used 3 + 2 = 5 apples. 12 - 5 = 7 apples remaining.",
            "answer": "7"
        }
    },
    {
        "inputs": {"problem": "A rectangle has length 8 and width 5. What is its area?"},
        "outputs": {
            "reasoning": "Area of rectangle = length × width = 8 × 5 = 40 square units.",
            "answer": "40"
        }
    },
    # ... more examples
]

# Success metric that considers both reasoning and final answer
def math_metric(predicted, expected):
    # Check if final answer is correct (most important)
    pred_answer = predicted.get("answer", "").strip()
    expected_answer = expected.get("answer", "").strip()
    answer_correct = pred_answer == expected_answer
    
    # Check if reasoning mentions key concepts
    reasoning = predicted.get("reasoning", "").lower()
    expected_reasoning = expected.get("reasoning", "").lower()
    
    # Look for mathematical operations/concepts
    math_keywords = ["add", "subtract", "multiply", "divide", "+", "-", "×", "÷", "="]
    reasoning_score = sum(1 for keyword in math_keywords 
                         if keyword in reasoning) / len(math_keywords)
    
    # Combined score (70% answer accuracy, 30% reasoning quality)
    return 0.7 * answer_correct + 0.3 * min(1.0, reasoning_score)

# Configure bootstrap optimizer for math problems
optimizer = BootstrapFewShot(
    metric=math_metric,
    max_bootstrapped_demos=6,        # Generate 6 good examples
    max_labeled_demos=4,             # Add up to 4 labeled examples if needed
    teacher_settings={
        "temperature": 1.0,          # Moderate creativity for math
        "max_tokens": 300            # Enough space for reasoning
    },
    
    # Enhanced configuration
    config=BootstrapFewShotConfig(
        max_rounds=3,                # Multiple rounds of generation
        initial_teacher_temperature=1.0,
        temperature_decay=0.85,      # Gradually become more focused
        min_temperature=0.4,
        
        # Rescue mode for math (often challenging)
        rescue_mode_threshold=0.15,  # Trigger if < 15% baseline
        rescue_initial_temperature=1.3,
        
        # Diversity to avoid repetitive examples
        use_diversity_scoring=True,
        diversity_weight=0.25,       # Moderate diversity preference
        
        metric_threshold=0.6         # Require 60% score to use as demo
    )
)

# Run bootstrap optimization
async def optimize_math_solver():
    print("Starting bootstrap optimization for math solver...")
    
    result = await optimizer.optimize(
        module=math_solver,
        dataset=dataset,
        validation_set=dataset[:20]
    )
    
    print(f"\nBootstrap optimization completed:")
    print(f"  Baseline → Final: {result.metadata['baseline_score']:.1%} → {result.best_score:.1%}")
    print(f"  Improvement: {result.improvement:.1%}")
    print(f"  Optimization time: {result.optimization_time:.1f}s")
    
    # Show demonstration details
    print(f"\nDemonstration Details:")
    print(f"  Generated {result.metadata['num_demos']} demonstrations")
    print(f"  Source: {result.metadata['num_bootstrapped']} bootstrapped, {result.metadata['num_labeled']} labeled")
    print(f"  Quality: {result.metadata['avg_demo_score']:.3f} avg score")
    
    # Show temperature progression
    temp_info = result.metadata['temperature_schedule']
    print(f"  Temperature: {temp_info['initial']:.2f} → {temp_info['final']:.2f}")
    print(f"  Rescue mode: {result.metadata['rescue_mode']}")
    
    return result.optimized_module

# Run optimization
optimized_math_solver = asyncio.run(optimize_math_solver())

# Test the optimized solver
test_problems = [
    "Tom has 15 baseball cards. He trades 4 cards for 6 new cards. How many cards does he have now?",
    "A square has sides of length 7 cm. What is its perimeter?",
    "Lisa buys 3 packages of stickers. Each package has 8 stickers. How many stickers does she have total?"
]

print(f"\nTesting optimized math solver:")
for problem in test_problems:
    result = await optimized_math_solver(problem=problem)
    print(f"\nProblem: {problem}")
    print(f"Reasoning: {result.outputs['reasoning']}")
    print(f"Answer: {result.outputs['answer']}")
```

## Performance Characteristics

Bootstrap Few-Shot performance depends on several factors:

### Dataset Size vs Demo Quality
- **Small datasets** (< 50 examples): May need more rounds to find successes
- **Large datasets** (> 500 examples): Can be selective about demo quality
- **Sweet spot**: 100-300 examples provides good balance

### Task Difficulty vs Teacher Temperature
- **Easy tasks**: Lower temperature (0.7-1.0) to avoid overthinking
- **Medium tasks**: Moderate temperature (1.0-1.3) for balanced exploration
- **Hard tasks**: Higher temperature (1.3-1.6) + rescue mode

### Success Threshold Tuning
```python
# Conservative (high-quality demos, fewer total)
config.metric_threshold = 0.8

# Balanced (default recommendation)  
config.metric_threshold = 0.7

# Permissive (more demos, lower average quality)
config.metric_threshold = 0.5
```

## Comparison with DSPy

| Feature | DSPy BootstrapFewShot | LogiLLM BootstrapFewShot |
|---------|----------------------|--------------------------|
| Basic teacher-student | ✅ | ✅ |
| Temperature scheduling | ❌ | ✅ |
| Rescue mode | ❌ | ✅ |
| Diversity-aware selection | ❌ | ✅ |
| Detailed analytics | Basic | Comprehensive |
| Configuration options | Limited | Extensive |
| Error handling | Basic | Robust |

LogiLLM's implementation provides much more control and sophisticated strategies for challenging optimization scenarios.

## Next Steps

- [Hybrid Optimizer](hybrid-optimizer.md) - Combine bootstrap with hyperparameter optimization
- [COPRO](copro.md) - Alternative instruction-focused optimization
- [SIMBA](simba.md) - More sophisticated prompt evolution with introspection
- [Format Optimizer](format-optimizer.md) - Optimize demonstration formats