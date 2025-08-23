# Your First LogiLLM Program

This tutorial walks through building a complete LogiLLM application step by step. We'll create a document summarization system that can be optimized for different use cases.

## What We'll Learn

- How signatures define program behavior
- How modules execute tasks
- How providers connect to LLMs
- How optimization improves performance
- How to handle errors and edge cases

## The Problem

We want to build a system that:
1. Takes long documents as input
2. Produces summaries of different types (brief, detailed, bullet points)
3. Can be optimized for accuracy vs. brevity
4. Handles various document types

## Step 1: Understanding the Core Concepts

**ELI5:** Imagine you're teaching an assistant to summarize books. Instead of giving them a script to follow word-for-word, you teach them the general pattern: read the book, identify key points, write a summary. They figure out the specific words to use.

**Technical:** LogiLLM separates the specification of what you want (signatures) from how to achieve it (modules) and where to run it (providers). This allows the same program to work with different LLMs and be optimized for different objectives.

## Step 2: Setting Up the Project

Create a new file `summarizer.py`:

```python
# Import core components
from logillm import Predict, ChainOfThought, Signature
from logillm import InputField, OutputField
from logillm.providers import create_provider, register_provider
from logillm.optimizers import HybridOptimizer

# Setup provider (we'll use mock for development)
from logillm.providers import MockProvider

# For development and testing
dev_provider = MockProvider()
register_provider(dev_provider, set_default=True)
```

## Step 3: Defining Signatures

Signatures are the heart of LogiLLM. They define what your program does:

```python
# Simple signature using string syntax
simple_summary = Predict("document -> summary")

# More detailed signature with types and descriptions
class SummarizationSignature(Signature):
    """Summarize documents with configurable detail level."""
    
    document: str = InputField(
        desc="The document to summarize",
        format_hint="Plain text or markdown"
    )
    
    style: str = InputField(
        desc="Summary style: 'brief', 'detailed', or 'bullets'",
        default="brief"
    )
    
    summary: str = OutputField(
        desc="The summary in the requested style"
    )
    
    key_points: list[str] = OutputField(
        desc="Main points extracted from document"
    )
    
    word_count: int = OutputField(
        desc="Approximate word count of summary"
    )
```

### How Signatures Work

**ELI5:** A signature is like a contract. It says "if you give me A, I'll give you B." The module figures out how to fulfill that contract.

**Technical:** Signatures use Python's type system and descriptors to define a validated interface. They can work with native Python types or Pydantic models for additional validation.

## Step 4: Creating Modules

Modules implement the execution strategy:

```python
# Basic prediction module
summarizer = Predict(signature=SummarizationSignature)

# Chain of thought for better reasoning
thoughtful_summarizer = ChainOfThought(
    signature=SummarizationSignature,
    reasoning_field="analysis"  # Add reasoning to output
)

# Using the module
document = """
The history of artificial intelligence (AI) began in antiquity, with myths, 
stories and rumors of artificial beings endowed with intelligence or consciousness 
by master craftsmen. The seeds of modern AI were planted by classical philosophers 
who attempted to describe the process of human thinking as the mechanical 
manipulation of symbols...
"""

result = summarizer(
    document=document,
    style="brief"
)

print(f"Summary: {result.summary}")
print(f"Key Points: {result.key_points}")
print(f"Word Count: {result.word_count}")
```

## Step 5: Advanced Module Composition

You can compose modules for complex tasks:

```python
from logillm import Module

class AdvancedSummarizer(Module):
    """Multi-stage summarization with quality checking."""
    
    def __init__(self):
        super().__init__()
        
        # First stage: Extract key information
        self.extractor = Predict("document -> main_ideas, supporting_details")
        
        # Second stage: Generate summary
        self.summarizer = ChainOfThought(SummarizationSignature)
        
        # Third stage: Quality check
        self.validator = Predict("""
            summary, original -> 
            is_accurate: bool,
            missing_points: list,
            quality_score: float
        """)
    
    async def forward(self, document: str, style: str = "brief"):
        # Extract key information
        extraction = await self.extractor(document=document)
        
        # Generate summary
        summary = await self.summarizer(
            document=document,
            style=style,
            context=extraction.main_ideas  # Pass context
        )
        
        # Validate quality
        validation = await self.validator(
            summary=summary.summary,
            original=document
        )
        
        # Return enhanced result
        return {
            **summary.dict(),
            "quality_score": validation.quality_score,
            "validation": validation
        }
```

## Step 6: Working with Providers

Providers connect to actual LLMs:

```python
# Development with mock provider
mock_provider = MockProvider(
    default_response="This is a mock summary for testing."
)

# Production with OpenAI
openai_provider = create_provider(
    "openai",
    model="gpt-4",
    temperature=0.7,
    max_tokens=500
)

# Register and use
register_provider(openai_provider, set_default=True)

# Same code works with different providers
result = summarizer(document=document, style="detailed")
```

### Provider Comparison

| Provider | When to Use | Configuration |
|----------|-------------|---------------|
| Mock | Development, testing | No API key needed |
| OpenAI | Production, GPT models | Requires OPENAI_API_KEY |
| Anthropic | Production, Claude models | Requires ANTHROPIC_API_KEY |
| Google | Production, Gemini models | Requires GOOGLE_API_KEY |

## Step 7: Optimization

This is where LogiLLM shines compared to other frameworks:

```python
from logillm.optimizers import HybridOptimizer, BootstrapFewShot

# Prepare training data
training_docs = [
    {
        "document": "Long article about climate change...",
        "summary": "Climate change poses significant risks...",
        "style": "brief"
    },
    # More examples...
]

# Method 1: Bootstrap few-shot learning
# Define accuracy metric
def accuracy_metric(prediction, reference):
    # Compare generated summary with expected summary
    return prediction.summary == reference.summary

bootstrap_opt = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=3)
optimized_summarizer = bootstrap_opt.compile(
    summarizer,
    dataset=training_docs
)

# Method 2: Hybrid optimization (LogiLLM's unique feature)
# Define success metric
def brevity_accuracy_metric(prediction, reference):
    accuracy = compare_summaries(prediction.summary, reference.summary)
    brevity = 1.0 - (len(prediction.summary) / len(reference.document))
    return 0.7 * accuracy + 0.3 * brevity

hybrid_opt = HybridOptimizer(
    metric=brevity_accuracy_metric,
    strategy="alternating",  # Optimize prompts and parameters
    optimize_format=True,    # Find best output format
    num_iterations=5
)

# Optimize
best_summarizer = hybrid_opt.optimize(
    module=summarizer,
    dataset=training_docs
)
```

### Understanding Optimization

**ELI5:** Optimization is like training. You show the system examples of good summaries, and it learns to produce similar quality outputs. LogiLLM can optimize both what it says (prompts) and how it says it (parameters like temperature).

**Technical:** LogiLLM's hybrid optimization simultaneously tunes prompts (instructions and examples) and model parameters (temperature, top_p, etc.). This is a unique capability not found in DSPy or similar frameworks.

## Step 8: Error Handling and Robustness

```python
from logillm import Retry, Refine
from logillm.exceptions import ModuleError, ProviderError

# Add retry logic for reliability
robust_summarizer = Retry(
    module=summarizer,
    max_retries=3,
    backoff_factor=2.0
)

# Add refinement for quality
refined_summarizer = Refine(
    module=summarizer,
    num_iterations=2,
    refinement_prompt="Improve clarity and conciseness"
)

# Handle errors gracefully
try:
    result = robust_summarizer(
        document=document,
        style="bullets"
    )
except ProviderError as e:
    print(f"LLM provider error: {e}")
    # Fall back to simpler approach
    result = simple_summary(document=document)
except ModuleError as e:
    print(f"Module execution error: {e}")
    print(f"Suggestions: {e.suggestions}")
```

## Step 9: Complete Application

Here's everything together:

```python
from logillm import ChainOfThought, Signature, InputField, OutputField
from logillm.providers import create_provider, register_provider
from logillm.optimizers import HybridOptimizer
from logillm import Retry

class DocumentSummarizer(Signature):
    """Professional document summarization."""
    
    document: str = InputField(desc="Document to summarize")
    style: str = InputField(desc="brief|detailed|bullets", default="brief")
    max_words: int = InputField(desc="Maximum words in summary", default=150)
    
    summary: str = OutputField(desc="The summary")
    key_points: list[str] = OutputField(desc="Main points")
    confidence: float = OutputField(desc="Confidence in summary quality")

def create_summarization_app(provider_name="openai", optimize=False):
    """Create a production-ready summarization application."""
    
    # Setup provider
    provider = create_provider(provider_name, model="gpt-4")
    register_provider(provider, set_default=True)
    
    # Create module with robustness
    base_module = ChainOfThought(signature=DocumentSummarizer)
    summarizer = Retry(base_module, max_retries=3)
    
    if optimize and training_data_available():
        # Define custom metric
        def custom_metric(prediction, reference):
            # Your custom evaluation logic here
            return 0.9  # Placeholder
        
        # Optimize for your use case
        optimizer = HybridOptimizer(metric=custom_metric, strategy="joint")
        summarizer = optimizer.optimize(
            summarizer,
            dataset=load_training_data()
        )
    
    return summarizer

# Use the application
if __name__ == "__main__":
    app = create_summarization_app(optimize=True)
    
    # Process documents
    documents = load_documents()  # Your documents
    
    for doc in documents:
        result = app(
            document=doc.content,
            style="brief",
            max_words=100
        )
        
        print(f"Document: {doc.title}")
        print(f"Summary: {result.summary}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Key Points:")
        for point in result.key_points:
            print(f"  - {point}")
        print("-" * 40)
```

## Key Takeaways

1. **Signatures define what you want** - They specify inputs and outputs without implementation details

2. **Modules implement how to do it** - Different modules (Predict, ChainOfThought) use different strategies

3. **Providers handle where it runs** - Same code works with different LLMs

4. **Optimization improves performance** - LogiLLM can optimize both prompts and parameters

5. **Composition enables complexity** - Combine simple modules into sophisticated applications

## Comparison with DSPy

If you're coming from DSPy, here's how this example would differ:

```python
# DSPy version
import dspy

class Summarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("document -> summary")
    
    def forward(self, document):
        return self.summarize(document=document)

# LogiLLM advantages:
# 1. No required dependencies (DSPy needs many)
# 2. Can optimize hyperparameters (DSPy cannot)
# 3. Cleaner error handling and types
# 4. Native async support
```

## Next Steps

- Explore [Modules](../modules/predict.md) for different execution strategies
- Learn about [Optimization](../optimization/overview.md) in depth
- Understand [Providers](../providers/openai.md) for production deployment
- Read [Advanced Topics](../advanced/callbacks.md) for monitoring and debugging

## Exercises

1. **Modify the style options**: Add a "technical" style that preserves technical terms

2. **Add language detection**: Extend the signature to detect and preserve the document's language

3. **Implement caching**: Add caching to avoid re-summarizing the same content

4. **Create a batch processor**: Modify the application to efficiently process multiple documents

5. **Add quality metrics**: Implement metrics to measure summary quality automatically