# Search Strategies: Zero-Dependency Optimization Algorithms

LogiLLM implements sophisticated optimization algorithms with zero external dependencies. These search strategies power the hyperparameter optimization in SIMBA, Hybrid Optimizer, and other advanced optimizers.

## Why Zero Dependencies?

Most optimization frameworks require heavy dependencies:
- **Optuna**: Large package with many dependencies
- **Scikit-optimize**: Requires NumPy, SciPy, matplotlib
- **Hyperopt**: MongoDB backend, complex setup

LogiLLM implements everything in pure Python, making it **2x faster** and eliminating dependency conflicts.

## Available Strategies

### 1. Random Search
Simple but surprisingly effective uniform sampling:

```python
from logillm.optimizers import RandomSearchStrategy, StrategyConfig

# Basic random search
strategy = RandomSearchStrategy(
    config=StrategyConfig(seed=42)
)

# Initialize with search space
search_space = SearchSpace({
    "temperature": ParamSpec(name="temperature", type=ParamType.FLOAT, range=(0.0, 2.0)),
    "top_p": ParamSpec(name="top_p", type=ParamType.FLOAT, range=(0.0, 1.0)),
    "max_tokens": ParamSpec(name="max_tokens", type=ParamType.INT, range=(50, 500))
})

strategy.initialize(search_space)

# Sample configurations
for trial in range(10):
    config = strategy.suggest_next()
    score = evaluate_config(config)
    strategy.update(config, score)
```

### 2. Grid Search
Systematic exploration of all parameter combinations:

```python
from logillm.optimizers import GridSearchStrategy

# Grid search with custom resolution
strategy = GridSearchStrategy(
    config=StrategyConfig(seed=42),
    resolution=5  # 5 points per continuous parameter
)

strategy.initialize(search_space)

# This creates:
# - temperature: [0.0, 0.5, 1.0, 1.5, 2.0]
# - top_p: [0.0, 0.25, 0.5, 0.75, 1.0]  
# - max_tokens: [50, 162, 275, 387, 500]
# Total: 5 × 5 × 5 = 125 configurations

for trial in range(125):
    config = strategy.suggest_next()
    score = evaluate_config(config)
    strategy.update(config, score)
```

### 3. Bayesian Optimization (SimpleBayesianStrategy)
Sophisticated optimization using Gaussian Process surrogate models:

```python
from logillm.optimizers import SimpleBayesianStrategy, AcquisitionType

# Bayesian optimization with custom settings
strategy = SimpleBayesianStrategy(
    config=StrategyConfig(
        n_warmup=5,  # Random exploration first
        exploration_weight=0.2,  # Balance exploration vs exploitation
        acquisition_type=AcquisitionType.EXPECTED_IMPROVEMENT
    ),
    kernel_bandwidth=0.15  # RBF kernel parameter
)
```

### 4. Latin Hypercube Sampling
Space-filling design for efficient coverage:

```python
from logillm.optimizers import LatinHypercubeStrategy

# Latin hypercube with 50 samples
strategy = LatinHypercubeStrategy(
    config=StrategyConfig(seed=42),
    n_samples=50
)

# Ensures good coverage of parameter space with fewer samples than grid search
```

## Bayesian Optimization Deep Dive

The SimpleBayesianStrategy is LogiLLM's most sophisticated algorithm:

### Gaussian Process Surrogate Model

```python
def predict(config):
    """Predict mean and uncertainty using kernel regression."""
    weights = []
    values = []
    
    # Compute kernel weights for all observations
    for obs_config, obs_score in observations:
        distance = config_distance(config, obs_config)
        weight = math.exp(-distance / (2 * kernel_bandwidth**2))  # RBF kernel
        weights.append(weight)
        values.append(obs_score)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Weighted mean prediction
    mean = sum(w * v for w, v in zip(weights, values))
    
    # Weighted variance (uncertainty)
    variance = sum(w * (v - mean)**2 for w, v in zip(weights, values))
    std = math.sqrt(variance) if variance > 0 else 0.1
    
    return mean, std
```

### Acquisition Functions

LogiLLM implements multiple acquisition functions for balancing exploration vs exploitation:

#### Expected Improvement
```python
def expected_improvement(mean, std, best_score):
    """Expected improvement over current best."""
    if std == 0:
        return 0.0
    
    z = (mean - best_score) / std
    ei = std * (z * normal_cdf(z) + normal_pdf(z))
    return ei + exploration_weight * std
```

#### Upper Confidence Bound  
```python
def upper_confidence_bound(mean, std):
    """Upper confidence bound (optimistic estimate)."""
    beta = 2.0  # Exploration parameter
    return mean + beta * std
```

#### Probability of Improvement
```python
def probability_of_improvement(mean, std, best_score):
    """Probability of improving over current best."""
    if std == 0:
        return 0.0
    
    z = (mean - best_score) / std
    return normal_cdf(z)
```

### Configuration Distance Metric

The Bayesian optimizer needs to measure similarity between configurations:

```python
def config_distance(config1, config2):
    """Compute normalized distance between configurations."""
    distance = 0.0
    n_params = 0
    
    for param_name in config1:
        if param_name not in config2:
            continue
            
        spec = search_space.param_specs[param_name]
        val1, val2 = config1[param_name], config2[param_name]
        
        if spec.param_type in ["float", "int"] and spec.range:
            # Normalized Euclidean distance for numeric parameters
            min_val, max_val = spec.range
            normalized_dist = abs(val1 - val2) / (max_val - min_val)
            distance += normalized_dist**2
            
        elif spec.param_type == "categorical":
            # Hamming distance for categorical parameters
            distance += (1.0 if val1 != val2 else 0.0)
            
        n_params += 1
    
    return math.sqrt(distance / max(1, n_params))
```

## Strategy Selection Guide

### Task Characteristics

| Task Type | Recommended Strategy | Reason |
|-----------|---------------------|---------|
| **Few trials** (< 20) | Latin Hypercube | Best coverage with limited budget |
| **Medium budget** (20-100) | Bayesian | Good balance of exploration/exploitation |
| **Large budget** (> 100) | Bayesian → Random | Start smart, then explore broadly |
| **Discrete parameters** | Grid Search | Systematic coverage of combinations |
| **High-dimensional** (> 10 params) | Random Search | Curse of dimensionality affects others |
| **Mixed types** | Bayesian | Handles continuous + categorical well |

### Performance Characteristics

```python
# Performance comparison on typical hyperparameter optimization

Strategy              | Convergence Speed | Final Quality | Computational Cost
---------------------|------------------|---------------|-------------------
Random               | Slow             | Good          | Very Low
Grid                 | Medium           | Excellent*    | High
Latin Hypercube      | Fast             | Very Good     | Low
Bayesian            | Very Fast        | Excellent     | Medium

# * Excellent only if resolution is high enough
```

## Advanced Configuration

### Acquisition Function Comparison

```python
from logillm.optimizers import StrategyConfig, AcquisitionType

# Expected Improvement (default) - balanced exploration/exploitation
ei_config = StrategyConfig(
    acquisition_type=AcquisitionType.EXPECTED_IMPROVEMENT,
    exploration_weight=0.1  # Lower = more exploitation
)

# Upper Confidence Bound - more exploration
ucb_config = StrategyConfig(
    acquisition_type=AcquisitionType.UPPER_CONFIDENCE_BOUND,
    exploration_weight=0.2  # Higher values increase exploration
)

# Probability of Improvement - conservative
poi_config = StrategyConfig(
    acquisition_type=AcquisitionType.PROBABILITY_OF_IMPROVEMENT
)
```

### Kernel Bandwidth Tuning

```python
# Kernel bandwidth affects how much nearby points influence predictions

# Narrow kernel (0.05-0.1) - Local optimization
# - Good for fine-tuning around known good regions
# - Risk of getting stuck in local optima

# Medium kernel (0.1-0.2) - Balanced (default)
# - Good general-purpose choice
# - Reasonable exploration/exploitation trade-off

# Wide kernel (0.2-0.5) - Global exploration  
# - Good for very rugged optimization landscapes
# - Slower convergence but more thorough exploration

strategy = SimpleBayesianStrategy(
    kernel_bandwidth=0.15,  # Tune based on your problem
    config=StrategyConfig(n_warmup=10)
)
```

## Real-World Example

```python
import asyncio
from logillm import Predict
from logillm.optimizers import HyperparameterOptimizer, SimpleBayesianStrategy
from logillm.core.parameters import SearchSpace, ParamSpec, ParamType

# Create a module to optimize
qa_module = Predict("question: str -> answer: str")

# Define hyperparameter search space
search_space = SearchSpace({
    "temperature": ParamSpec(
        name="temperature", 
        param_type=ParamType.FLOAT,
        range=(0.0, 1.5),
        default=0.7,
        description="Sampling temperature"
    ),
    "top_p": ParamSpec(
        name="top_p",
        param_type=ParamType.FLOAT, 
        range=(0.1, 1.0),
        default=0.9,
        description="Nucleus sampling threshold"
    ),
    "max_tokens": ParamSpec(
        name="max_tokens",
        param_type=ParamType.INT,
        range=(50, 300),
        default=150,
        description="Maximum response length"
    ),
    "response_style": ParamSpec(
        name="response_style",
        param_type=ParamType.CATEGORICAL,
        choices=["concise", "detailed", "step_by_step"],
        default="concise",
        description="Response style preference"
    )
})

# Compare different search strategies
async def compare_strategies():
    dataset = load_qa_dataset()
    
    def accuracy_metric(predicted, expected):
        return float(predicted.get("answer", "").lower().strip() == 
                    expected.get("answer", "").lower().strip())
    
    strategies = {
        "random": RandomSearchStrategy(StrategyConfig(seed=42)),
        "bayesian_ei": SimpleBayesianStrategy(
            StrategyConfig(
                seed=42,
                acquisition_type=AcquisitionType.EXPECTED_IMPROVEMENT,
                exploration_weight=0.1
            )
        ),
        "bayesian_ucb": SimpleBayesianStrategy(
            StrategyConfig(
                seed=42,
                acquisition_type=AcquisitionType.UPPER_CONFIDENCE_BOUND,
                exploration_weight=0.2
            )
        ),
        "latin_hypercube": LatinHypercubeStrategy(
            StrategyConfig(seed=42),
            n_samples=50
        )
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"Testing {name} strategy...")
        
        optimizer = HyperparameterOptimizer(
            metric=accuracy_metric,
            strategy=strategy,
            n_trials=30
        )
        
        result = await optimizer.optimize(qa_module, dataset)
        
        results[name] = {
            "best_score": result.best_score,
            "improvement": result.improvement,
            "optimization_time": result.optimization_time,
            "convergence": result.metadata.get("convergence_iteration", 30)
        }
        
        print(f"  Best score: {result.best_score:.3f}")
        print(f"  Improvement: {result.improvement:.3f}")
        print(f"  Time: {result.optimization_time:.1f}s")
    
    # Compare results
    print("\nStrategy Comparison:")
    print("Strategy         | Score  | Improve | Time  | Converge")
    print("-----------------|--------|---------|-------|----------")
    for name, metrics in results.items():
        print(f"{name:<15} | {metrics['best_score']:.3f} | {metrics['improvement']:+.3f} | {metrics['optimization_time']:5.1f}s | {metrics['convergence']:2d}")
    
    return results

# Run comparison
comparison_results = asyncio.run(compare_strategies())
```

## Strategy Factory

LogiLLM provides a factory function for easy strategy creation:

```python
from logillm.optimizers import create_strategy

# Create strategies by name
random_strategy = create_strategy("random", seed=42)
grid_strategy = create_strategy("grid", resolution=8)
bayesian_strategy = create_strategy("bayesian", kernel_bandwidth=0.12)
lhs_strategy = create_strategy("latin_hypercube", n_samples=75)

# Use in optimizers
optimizer = HyperparameterOptimizer(
    metric=my_metric,
    strategy=bayesian_strategy,  # or "bayesian" string
    n_trials=40
)
```

## Custom Strategy Implementation

You can implement custom search strategies by extending the base class:

```python
from logillm.optimizers import SearchStrategy

class SimulatedAnnealingStrategy(SearchStrategy):
    """Custom simulated annealing strategy."""
    
    def __init__(self, config=None, initial_temp=1.0, cooling_rate=0.95):
        super().__init__(config)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.current_temp = initial_temp
        self.current_config = None
        self.current_score = float("-inf")
    
    def _on_initialize(self):
        """Initialize with random configuration."""
        self.current_config = self.search_space.sample(self.rng)
        self.current_temp = self.initial_temp
    
    def suggest_next(self, history=None):
        """Suggest next configuration using SA."""
        if self.current_config is None:
            return self.search_space.sample(self.rng)
        
        # Generate neighbor configuration
        neighbor = self._generate_neighbor(self.current_config)
        return neighbor
    
    def update(self, config, score, metadata=None):
        """Update SA state."""
        if self.current_config is None:
            self.current_config = config
            self.current_score = score
            return
        
        # Accept or reject based on SA criteria
        if score > self.current_score:
            # Always accept improvement
            self.current_config = config
            self.current_score = score
        else:
            # Accept worse solutions with probability
            delta = score - self.current_score
            prob = math.exp(delta / self.current_temp)
            if self.rng.random() < prob:
                self.current_config = config
                self.current_score = score
        
        # Cool down
        self.current_temp *= self.cooling_rate
        self.iteration += 1
    
    def _generate_neighbor(self, config):
        """Generate neighboring configuration."""
        # Implement neighbor generation logic
        # ...
        pass
    
    @property
    def name(self):
        return "simulated_annealing"
    
    def _on_reset(self):
        self.current_config = None
        self.current_score = float("-inf")  
        self.current_temp = self.initial_temp
```

## Next Steps

- [Hybrid Optimizer](hybrid-optimizer.md) - Uses these strategies for joint optimization
- [SIMBA](simba.md) - Advanced optimizer using Bayesian strategies
- [Overview](overview.md) - Complete LogiLLM optimization system
- [Bootstrap Few-Shot](bootstrap-fewshot.md) - Demonstration-based optimization