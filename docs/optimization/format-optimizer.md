# Format Optimizer: Automatic Format Discovery

The Format Optimizer is LogiLLM's unique capability to automatically discover the optimal prompt format for any task and model combination. It treats format as a hyperparameter that can be optimized alongside prompts and other settings.

## Why Format Matters

Different LLMs and tasks perform better with different prompt formats:

- **GPT models** often prefer structured JSON for complex outputs
- **Claude models** may work better with XML-style prompts  
- **Mathematical tasks** benefit from step-by-step Markdown formatting
- **Classification tasks** work well with clean categorical JSON
- **Reasoning tasks** may need hybrid approaches with thinking tags

The Format Optimizer automatically tests these formats to find what works best.

## Available Formats

LogiLLM tests several prompt formats systematically:

### 1. Markdown Format
Natural language with Markdown structure:
```markdown
# Task
Analyze the given text and provide a sentiment classification.

## Input
**text**: "I love this product! It works perfectly."

## Output
Please provide your response in the following format:

**sentiment**: positive/negative/neutral
**confidence**: 0.0 to 1.0
**reasoning**: Brief explanation
```

### 2. JSON Format  
Structured JSON input and output:
```json
{
  "task": "Analyze sentiment of the given text",
  "input": {
    "text": "I love this product! It works perfectly."
  },
  "output_format": {
    "sentiment": "string (positive/negative/neutral)",
    "confidence": "number (0.0 to 1.0)",
    "reasoning": "string (brief explanation)"
  }
}
```

### 3. XML Format
XML-structured with clear boundaries:
```xml
<task>
  <instruction>Analyze the given text and provide sentiment classification</instruction>
  <input>
    <text>I love this product! It works perfectly.</text>
  </input>
  <output_schema>
    <sentiment>positive/negative/neutral</sentiment>
    <confidence>0.0 to 1.0</confidence>
    <reasoning>brief explanation</reasoning>
  </output_schema>
</task>
```

### 4. Hybrid Formats (Coming Soon)
- **Hybrid MD-JSON**: Markdown structure with JSON output
- **Hybrid XML-JSON**: XML input with JSON output  
- **Cognitive**: Adaptive format based on task type

## Basic Usage

```python
from logillm.optimizers import FormatOptimizer, PromptFormat
import asyncio

# Define success metric
def accuracy_metric(predicted, expected):
    return float(predicted.get("sentiment", "").lower() == 
                expected.get("sentiment", "").lower())

# Create format optimizer
optimizer = FormatOptimizer(
    metric=accuracy_metric,
    track_by_model=True  # Learn format preferences per model
)

# Find optimal format
result = await optimizer.optimize(
    module=sentiment_classifier,
    dataset=training_data,
    validation_set=test_data
)

print(f"Best format: {result.metadata['best_format']}")
print(f"Format scores: {result.metadata['format_scores']}")
print(f"Improvement: {result.improvement:.2%}")
```

## Configuration Options

The Format Optimizer supports extensive configuration:

```python
from logillm.optimizers import FormatOptimizer, FormatOptimizerConfig, PromptFormat

config = FormatOptimizerConfig(
    formats_to_test=[
        PromptFormat.MARKDOWN,
        PromptFormat.JSON, 
        PromptFormat.XML,
        # PromptFormat.HYBRID_MD_JSON,  # Coming soon
    ],
    min_samples_per_format=5,      # Minimum evaluations per format
    max_samples_per_format=20,     # Maximum evaluations per format
    early_stopping_threshold=0.2,  # Stop if format is 20% worse
    adaptive_sampling=True,        # Allocate more samples to promising formats
    
    # Multi-factor scoring
    consider_latency=True,         # Factor in response time
    consider_stability=True,       # Factor in score variance
    latency_weight=0.1,           # Weight for latency factor
    stability_weight=0.2          # Weight for stability factor
)

optimizer = FormatOptimizer(metric=accuracy_metric, config=config)
```

## The Format Testing Algorithm

### Step 1: Format Application

Each format is applied to the module by setting the appropriate adapter:

```python
def apply_format(module, format_type):
    formatted_module = module.deepcopy()
    
    if format_type == PromptFormat.MARKDOWN:
        formatted_module.adapter = MarkdownAdapter()
    elif format_type == PromptFormat.JSON:
        formatted_module.adapter = JSONAdapter()
    elif format_type == PromptFormat.XML:
        formatted_module.adapter = XMLAdapter()
    # ... other formats
    
    return formatted_module
```

### Step 2: Adaptive Sampling

The optimizer uses adaptive sampling to focus on promising formats:

```python
async def adaptive_evaluate(module, dataset, format_type, model_id):
    performance = format_performance[model_id][format_type]
    
    # Start with minimum samples
    n_samples = config.min_samples_per_format
    
    for i in range(config.max_samples_per_format):
        if i >= n_samples:
            # Check if we should continue sampling this format
            if not should_continue_sampling(performance, model_id):
                break
        
        # Evaluate on next sample
        sample = dataset[i % len(dataset)]
        start_time = time.time()
        
        try:
            score, _ = await evaluate(module, [sample])
            latency = time.time() - start_time
            
            performance.scores.append(score)
            performance.latencies.append(latency)
            performance.total_attempts += 1
            
        except Exception as e:
            # Track parsing failures
            performance.parse_failures += 1
            performance.total_attempts += 1
            performance.scores.append(0.0)
    
    return calculate_weighted_score(performance)
```

### Step 3: Multi-Factor Scoring

The optimizer considers multiple factors beyond accuracy:

```python
def calculate_weighted_score(performance):
    # Base score: accuracy × success rate
    score = performance.mean_score * performance.success_rate
    
    # Factor in latency (lower is better)
    if config.consider_latency:
        latency_factor = 1.0 / (1.0 + performance.mean_latency)
        score = (score * (1 - config.latency_weight) + 
                latency_factor * config.latency_weight)
    
    # Factor in stability (lower variance is better)
    if config.consider_stability:
        stability = 1.0 / (1.0 + variance(performance.scores))
        score = (score * (1 - config.stability_weight) + 
                stability * config.stability_weight)
    
    return score
```

## Performance Tracking

The optimizer tracks detailed performance metrics for each format:

```python
@dataclass
class FormatPerformance:
    format: PromptFormat
    scores: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    parse_failures: int = 0
    total_attempts: int = 0
    
    @property
    def success_rate(self) -> float:
        """Parsing success rate."""
        if self.total_attempts == 0:
            return 0.0
        return 1.0 - (self.parse_failures / self.total_attempts)
    
    @property
    def mean_score(self) -> float:
        """Average score across successful runs."""
        return statistics.mean(self.scores) if self.scores else 0.0
    
    @property
    def stability(self) -> float:
        """Score stability (inverse of variance)."""
        if len(self.scores) < 2:
            return 0.0
        return 1.0 / (1.0 + statistics.variance(self.scores))
```

## Real-World Example

```python
import asyncio
from logillm import Predict
from logillm.optimizers import FormatOptimizer, FormatOptimizerConfig

# Create a data extraction module
extractor = Predict("""
document: str -> entities: list[str], categories: list[str], summary: str
""")

# Dataset with document extraction tasks
dataset = [
    {
        "inputs": {
            "document": "Apple Inc. reported Q3 earnings of $1.2B. CEO Tim Cook praised the iPhone sales in Europe and Asia."
        },
        "outputs": {
            "entities": ["Apple Inc.", "Tim Cook", "iPhone", "Europe", "Asia"],
            "categories": ["company", "earnings", "product", "geography"],
            "summary": "Apple reported strong Q3 earnings with good iPhone performance internationally."
        }
    },
    # ... more examples
]

# Multi-factor success metric for extraction
def extraction_metric(predicted, expected):
    # Entity extraction accuracy
    pred_entities = set(predicted.get("entities", []))
    expected_entities = set(expected.get("entities", []))
    
    if expected_entities:
        entity_recall = len(pred_entities & expected_entities) / len(expected_entities)
        entity_precision = len(pred_entities & expected_entities) / len(pred_entities) if pred_entities else 0
        entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
    else:
        entity_f1 = 1.0 if not pred_entities else 0.0
    
    # Category accuracy  
    pred_categories = set(predicted.get("categories", []))
    expected_categories = set(expected.get("categories", []))
    category_jaccard = (len(pred_categories & expected_categories) / 
                       len(pred_categories | expected_categories)) if (pred_categories | expected_categories) else 1.0
    
    # Summary completeness (simple keyword overlap)
    pred_summary = predicted.get("summary", "").lower().split()
    expected_summary = expected.get("summary", "").lower().split()
    summary_overlap = len(set(pred_summary) & set(expected_summary)) / len(set(expected_summary)) if expected_summary else 1.0
    
    # Combined score (40% entities, 30% categories, 30% summary)
    return 0.4 * entity_f1 + 0.3 * category_jaccard + 0.3 * summary_overlap

# Configure format optimizer for extraction
config = FormatOptimizerConfig(
    formats_to_test=[
        PromptFormat.MARKDOWN,  # Natural language descriptions
        PromptFormat.JSON,      # Structured data format
        PromptFormat.XML        # Clear field boundaries
    ],
    min_samples_per_format=8,      # More samples for complex task
    max_samples_per_format=25,     # Allow thorough testing
    adaptive_sampling=True,        # Focus on promising formats
    
    # Consider multiple factors for extraction
    consider_latency=True,         # Extraction can be slow
    consider_stability=True,       # Want consistent results
    latency_weight=0.15,          # Moderate latency concern
    stability_weight=0.25,        # High stability concern
    
    early_stopping_threshold=0.25  # More lenient for complex task
)

optimizer = FormatOptimizer(
    metric=extraction_metric,
    config=config,
    track_by_model=True  # Learn model-specific preferences
)

# Run format optimization
async def optimize_extraction_format():
    print("Starting format optimization for document extraction...")
    
    result = await optimizer.optimize(
        module=extractor,
        dataset=dataset,
        validation_set=dataset[:15]
    )
    
    print(f"\nFormat optimization completed:")
    print(f"  Best format: {result.metadata['best_format']}")
    print(f"  Best score: {result.best_score:.3f}")
    print(f"  Improvement: {result.improvement:.3f}")
    
    # Show format comparison
    print(f"\nFormat Performance Comparison:")
    format_scores = result.metadata['format_scores']
    for fmt, score in sorted(format_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fmt}: {score:.3f}")
    
    # Show detailed performance metrics
    performance_summary = result.metadata['format_performance']
    print(f"\nDetailed Performance Metrics:")
    for fmt, metrics in performance_summary.items():
        print(f"  {fmt}:")
        print(f"    Mean score: {metrics['mean_score']:.3f}")
        print(f"    Success rate: {metrics['success_rate']:.2%}")
        print(f"    Mean latency: {metrics['mean_latency']:.3f}s")
        print(f"    Stability: {metrics['stability']:.3f}")
        print(f"    Samples: {metrics['samples']}")
    
    return result.optimized_module

# Run optimization
optimized_extractor = asyncio.run(optimize_extraction_format())

# Test with the optimal format
test_document = """
Microsoft Corporation announced record revenue of $5.8B for Q2. 
Satya Nadella highlighted strong growth in Azure cloud services across 
North America and European markets.
"""

print(f"\nTesting optimized extractor:")
result = await optimized_extractor(document=test_document)
print(f"Entities: {result.outputs['entities']}")
print(f"Categories: {result.outputs['categories']}")
print(f"Summary: {result.outputs['summary']}")
```

## Format Recommendations

The optimizer learns format preferences and provides recommendations:

```python
# Get learned format preferences
recommendations = optimizer.get_format_recommendations()

print("Format Recommendations by Model:")
for model_id, best_format in recommendations["by_model"]:
    print(f"  {model_id}: {best_format.value}")

print("\nOpenAI Models:")
for model_id, format in recommendations["openai_models"]:
    print(f"  {model_id}: {format.value}")

print("\nAnthropic Models:")  
for model_id, format in recommendations["anthropic_models"]:
    print(f"  {model_id}: {format.value}")
```

## Integration with Hybrid Optimizer

Format optimization works seamlessly with the Hybrid Optimizer:

```python
from logillm.optimizers import HybridOptimizer

# Hybrid optimization including format
hybrid_optimizer = HybridOptimizer(
    metric=extraction_metric,
    strategy="alternating",
    optimize_format=True,  # Enable format optimization
    num_iterations=4
)

# This will optimize:
# 1. Format (JSON vs XML vs Markdown)
# 2. Hyperparameters (temperature, top_p, etc.)
# 3. Prompts (instructions, demonstrations)
result = await hybrid_optimizer.optimize(extractor, dataset)

print(f"Best format: {result.metadata.get('best_format')}")
print(f"Total improvement: {result.improvement:.2%}")
```

## Performance Characteristics

Format optimization results vary by task and model:

### Typical Performance Patterns

| Task Type | Best Format | Improvement | Notes |
|-----------|-------------|-------------|-------|
| Classification | JSON | 15-25% | Clear categorical outputs |
| Extraction | XML | 20-35% | Field boundaries help parsing |
| Reasoning | Markdown | 10-20% | Natural step-by-step flow |
| Code Generation | JSON | 25-40% | Structured templates work well |
| Creative Writing | Markdown | 5-15% | Natural language preferred |

### Model-Specific Patterns

- **GPT-4**: Often prefers JSON for structured tasks, Markdown for reasoning
- **GPT-3.5**: XML can help with complex extractions where JSON fails
- **Claude**: Strong performance across formats, slight XML preference
- **Local models**: May need simpler formats with more explicit structure

## Next Steps

- [Hybrid Optimizer](hybrid-optimizer.md) - Combine format with prompt+hyperparameter optimization
- [Search Strategies](strategies.md) - The optimization algorithms underlying format testing
- [Overview](overview.md) - Complete LogiLLM optimization system
- [COPRO](copro.md) - Instruction optimization to complement format optimization