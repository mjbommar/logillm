# LogiLLM Optimization Documentation

Welcome to LogiLLM's optimization documentation. This is LogiLLM's **key competitive advantage** - the ability to optimize both prompts and hyperparameters simultaneously, something that's fundamentally impossible in DSPy.

## 🚀 Quick Start

```python
from logillm import Predict
from logillm.optimizers import HybridOptimizer, accuracy_metric

# Create module
qa = Predict("question -> answer")

# Create hybrid optimizer (LogiLLM's killer feature)  
optimizer = HybridOptimizer(
    metric=accuracy_metric,
    strategy="alternating",     # alternating, joint, sequential
    optimize_format=True        # Also test JSON vs XML vs Markdown
)

# Optimize everything together
result = await optimizer.optimize(qa, dataset, validation_set)
print(f"Improvement: {result.improvement:.2%}")
```

## 📚 Documentation Index

### Core Concepts
- **[Overview](overview.md)** - The three types of optimization and why LogiLLM beats DSPy
- **[Hybrid Optimizer](hybrid-optimizer.md)** - LogiLLM's killer feature that DSPy cannot match

### Optimization Algorithms  
- **[SIMBA](simba.md)** - Stochastic introspective mini-batch ascent with zero dependencies
- **[COPRO](copro.md)** - Collaborative prompt optimization through iterative refinement
- **[Bootstrap Few-Shot](bootstrap-fewshot.md)** - Teacher-student demonstration generation
- **[Format Optimizer](format-optimizer.md)** - Automatic discovery of optimal prompt formats
- **[Search Strategies](strategies.md)** - Bayesian, grid, random, and Latin hypercube algorithms

## 🏆 LogiLLM vs DSPy Comparison

| Feature | DSPy | LogiLLM |
|---------|------|---------|
| **Prompt Optimization** | ✅ | ✅ |
| **Hyperparameter Optimization** | ❌ | ✅ |
| **Joint Optimization** | ❌ | ✅ |
| **Format Optimization** | ❌ | ✅ |
| **Zero Dependencies** | ❌ | ✅ |
| **Multiple Strategies** | ❌ | ✅ |
| **Temperature Scheduling** | ❌ | ✅ |
| **Rescue Mode** | ❌ | ✅ |

## 🎯 Optimization Strategy Guide

### By Task Complexity
- **Simple tasks** (classification): [Format Optimizer](format-optimizer.md) + [COPRO](copro.md)
- **Medium tasks** (Q&A): [Bootstrap Few-Shot](bootstrap-fewshot.md) + [Hybrid Optimizer](hybrid-optimizer.md)
- **Complex tasks** (reasoning): [SIMBA](simba.md) + [Hybrid Optimizer](hybrid-optimizer.md) with joint strategy

### By Compute Budget
- **Limited budget** (< 50 trials): [Hybrid Optimizer](hybrid-optimizer.md) sequential strategy
- **Medium budget** (50-200 trials): [Hybrid Optimizer](hybrid-optimizer.md) alternating strategy  
- **Large budget** (> 200 trials): [Hybrid Optimizer](hybrid-optimizer.md) joint strategy + [SIMBA](simba.md)

### By Performance Goals
- **Quick wins** (15-25% improvement): [Bootstrap Few-Shot](bootstrap-fewshot.md) alone
- **Strong performance** (25-35% improvement): [Hybrid Optimizer](hybrid-optimizer.md) alternating
- **Maximum performance** (35%+ improvement): [Hybrid Optimizer](hybrid-optimizer.md) joint + [Format Optimizer](format-optimizer.md)

## 🔧 Common Patterns

### Standard Optimization Pipeline
```python
from logillm.optimizers import HybridOptimizer, BootstrapFewShot, COPRO

# 1. Start with bootstrap for demonstrations
bootstrap = BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=4)
bootstrap_result = await bootstrap.optimize(module, dataset)

# 2. Refine instructions with COPRO
copro = COPRO(metric=my_metric, breadth=6, depth=3)
copro_result = await copro.optimize(bootstrap_result.optimized_module, dataset)

# 3. Final hybrid optimization
hybrid = HybridOptimizer(
    metric=my_metric,
    strategy="alternating",
    optimize_format=True
)
final_result = await hybrid.optimize(copro_result.optimized_module, dataset)
```

### One-Shot Optimization
```python
# Everything in one optimizer (recommended)
optimizer = HybridOptimizer(
    metric=my_metric,
    strategy="joint",           # Optimize everything simultaneously
    optimize_format=True,       # Include format optimization
    prompt_optimizer=SIMBA(     # Use SIMBA for sophisticated prompt evolution
        metric=my_metric,
        max_steps=8,
        num_candidates=6
    )
)

result = await optimizer.optimize(module, dataset, validation_set)
```

### Research and Experimentation
```python
# Try different strategies and compare
strategies = ["alternating", "joint", "sequential"]
results = {}

for strategy in strategies:
    optimizer = HybridOptimizer(
        metric=my_metric,
        strategy=strategy,
        optimize_format=True
    )
    result = await optimizer.optimize(module, dataset)
    results[strategy] = result.improvement

best_strategy = max(results.keys(), key=lambda k: results[k])
print(f"Best strategy: {best_strategy} ({results[best_strategy]:.2%} improvement)")
```

## 📊 Performance Expectations

### Typical Improvements by Optimizer

| Optimizer | Improvement Range | Best For |
|-----------|-------------------|----------|
| [Bootstrap Few-Shot](bootstrap-fewshot.md) | 15-30% | Tasks needing examples |
| [COPRO](copro.md) | 10-25% | Instruction-sensitive tasks |
| [Format Optimizer](format-optimizer.md) | 5-20% | Parsing-sensitive tasks |
| [SIMBA](simba.md) | 20-35% | Complex reasoning tasks |
| [Hybrid Optimizer](hybrid-optimizer.md) | 25-45% | All tasks (combines multiple approaches) |

### Optimization Time by Strategy

| Strategy | Time per Trial | Total Time | Use Case |
|----------|----------------|------------|----------|
| Sequential | 2x baseline | Fast | Quick wins |
| Alternating | 3-5x baseline | Medium | Balanced approach |
| Joint | 5-10x baseline | Slower | Maximum performance |

## 🛠️ Advanced Topics

### Custom Metrics
```python
def custom_metric(predicted, expected):
    # Accuracy component
    accuracy = float(predicted.get("answer", "").lower() == 
                    expected.get("answer", "").lower())
    
    # Confidence calibration component
    confidence = predicted.get("confidence", 0.5)
    calibration = 1.0 - abs(confidence - accuracy)
    
    # Latency component (if available)
    latency_penalty = 0.0
    if "latency" in predicted:
        latency_penalty = min(0.2, predicted["latency"] / 10.0)
    
    # Combined score
    return 0.6 * accuracy + 0.3 * calibration - 0.1 * latency_penalty
```

### Multi-Objective Optimization
```python
from logillm.optimizers import MultiObjectiveOptimizer

# Optimize for accuracy, speed, and cost simultaneously
optimizer = MultiObjectiveOptimizer(
    objectives={
        "accuracy": lambda pred, exp: accuracy_metric(pred, exp),
        "speed": lambda pred, exp: 1.0 / pred.get("latency", 1.0),
        "cost": lambda pred, exp: 1.0 / pred.get("cost", 1.0)
    },
    weights={"accuracy": 0.5, "speed": 0.3, "cost": 0.2}
)
```

### Optimization Callbacks
```python
from logillm.core.callbacks import CallbackManager

def optimization_callback(iteration, score, config, metadata):
    print(f"Iteration {iteration}: score={score:.3f}")
    if score > 0.9:
        print("🎉 Reached 90% accuracy!")
        return True  # Stop optimization
    return False

# Use callbacks during optimization
with CallbackManager() as cm:
    cm.add_callback("optimization_step", optimization_callback)
    
    result = await optimizer.optimize(module, dataset)
```

## 🔍 Debugging and Monitoring

### Optimization Analysis
```python
# Analyze optimization results
result = await optimizer.optimize(module, dataset)

print(f"Optimization Summary:")
print(f"  Strategy: {result.metadata.get('strategy')}")
print(f"  Iterations: {result.iterations}")
print(f"  Best score: {result.best_score:.3f}")
print(f"  Improvement: {result.improvement:.3f}")
print(f"  Time: {result.optimization_time:.1f}s")

# Check convergence
if "score_trajectory" in result.metadata:
    scores = result.metadata["score_trajectory"]
    print(f"  Score trajectory: {[f'{s:.3f}' for s in scores[-5:]]}")
    
    # Plot trajectory (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        plt.plot(scores)
        plt.title("Optimization Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.show()
    except ImportError:
        pass
```

### Performance Profiling
```python
import time
from logillm.core.usage_tracker import track_usage

# Profile optimization performance
with track_usage() as tracker:
    start_time = time.time()
    result = await optimizer.optimize(module, dataset)
    total_time = time.time() - start_time

usage = tracker.get_usage()
print(f"Optimization Profiling:")
print(f"  Total time: {total_time:.1f}s")
print(f"  API calls: {usage.total_requests}")
print(f"  Tokens: {usage.total_tokens}")
print(f"  Cost: ${usage.total_cost:.3f}")
print(f"  Time per trial: {total_time/result.iterations:.1f}s")
```

## 🚀 Getting Started

1. **Start simple**: Try [Bootstrap Few-Shot](bootstrap-fewshot.md) on your task
2. **Add instructions**: Use [COPRO](copro.md) to refine prompts  
3. **Optimize format**: Apply [Format Optimizer](format-optimizer.md)
4. **Go hybrid**: Use [Hybrid Optimizer](hybrid-optimizer.md) for maximum performance
5. **Advanced scenarios**: Try [SIMBA](simba.md) for complex reasoning tasks

## 📖 Further Reading

- **[Core Concepts](../core-concepts/)** - Understand modules, signatures, and providers
- **[Providers](../providers/)** - Configure OpenAI, Anthropic, and other LLM providers  
- **[API Reference](../api-reference/)** - Complete API documentation
- **[Examples](../../examples/)** - Real-world optimization examples

---

**💡 Pro Tip**: LogiLLM's optimization system is designed to be **composable**. You can chain optimizers, use them as components in larger systems, or create custom optimization pipelines tailored to your specific needs. The zero-dependency architecture ensures everything works together seamlessly.