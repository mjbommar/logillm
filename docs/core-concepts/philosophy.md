# LogiLLM Philosophy: Programming vs Prompting

This document explains the fundamental paradigm shift that LogiLLM represents: moving from writing prompts to writing programs.

## The Problem with Prompting

Traditional LLM interaction relies on carefully crafted prompt strings:

```python
# Traditional approach: Brittle prompt engineering
prompt = """You are a helpful assistant. Given a question, provide a concise answer.

Question: {question}
Answer:"""

response = llm.complete(prompt.format(question="What is Python?"))
# Hope it works! What if format changes? What if we need different styles?
```

**Problems with this approach:**
- **Fragility**: Small prompt changes can break everything
- **No composability**: Can't easily combine prompts
- **No optimization**: Manual trial and error
- **No abstraction**: Implementation details everywhere
- **Poor maintainability**: Prompts scattered throughout code

## The Programming Paradigm

LogiLLM treats LLM interactions as programs with clear interfaces:

```python
# LogiLLM approach: Robust programming
from logillm import Predict

qa = Predict("question -> answer")  # Define what you want
result = qa(question="What is Python?")  # System figures out how
```

**Benefits of programming:**
- **Robustness**: Changes to implementation don't break interface
- **Composability**: Combine modules like functions
- **Optimization**: Automatically improve performance
- **Abstraction**: Hide complexity behind clean APIs
- **Maintainability**: Reusable, testable components

## Core Principles

### 1. Separation of Concerns

**ELI5:** Like separating a recipe (what to make) from cooking techniques (how to make it). You can improve your technique without changing the recipe.

**Technical:** LogiLLM separates:
- **What** (Signatures): Input/output specifications
- **How** (Modules): Execution strategies
- **Where** (Providers): LLM implementations
- **Optimization** (Optimizers): Performance improvement

```python
# WHAT: Signature defines the contract
class SentimentSignature(Signature):
    text: str = InputField()
    sentiment: str = OutputField()  # positive, negative, neutral
    confidence: float = OutputField()

# HOW: Module implements the strategy
analyzer = ChainOfThought(signature=SentimentSignature)

# WHERE: Provider handles LLM communication
provider = create_provider("openai", model="gpt-4")

# USE: Clean interface hides complexity
result = analyzer(text="I love this product!")
```

### 2. Declarative Over Imperative

**Traditional (Imperative):**
```python
# You specify exactly HOW to do something
def analyze_sentiment(text):
    prompt = f"Analyze the sentiment of: {text}\nProvide sentiment and confidence."
    response = openai.complete(prompt)
    # Parse response manually...
    return parsed_result
```

**LogiLLM (Declarative):**
```python
# You specify WHAT you want
analyzer = Predict("text -> sentiment, confidence: float")
# System determines how to achieve it
```

### 3. Optimization as a First-Class Citizen

Unlike traditional approaches where optimization is manual:

```python
# Traditional: Manual prompt engineering
prompts = [
    "Analyze sentiment:",
    "Determine if this is positive or negative:",
    "What is the sentiment of:",
    # Try different prompts manually...
]
```

LogiLLM makes optimization automatic:

```python
# Define accuracy metric
def accuracy_metric(prediction, reference):
    return prediction.sentiment == reference.sentiment

# LogiLLM: Automatic optimization
optimizer = HybridOptimizer(metric=accuracy_metric)
optimized_analyzer = optimizer.optimize(
    analyzer,
    dataset=training_data
)
# Automatically finds best prompts AND parameters
```

### 4. Composability

Modules compose like functions:

```python
# Compose simple modules into complex systems
class AnalysisPipeline(Module):
    def __init__(self):
        self.extract = Predict("document -> key_facts")
        self.analyze = ChainOfThought("facts -> analysis")
        self.summarize = Predict("analysis -> summary")
    
    async def forward(self, document):
        facts = await self.extract(document=document)
        analysis = await self.analyze(facts=facts.key_facts)
        summary = await self.summarize(analysis=analysis.analysis)
        return summary
```

## Comparison: Prompting vs Programming

| Aspect | Prompting | Programming (LogiLLM) |
|--------|-----------|----------------------|
| **Abstraction** | Low - prompts everywhere | High - clean interfaces |
| **Reusability** | Copy-paste prompts | Composable modules |
| **Testing** | Test full prompts | Test individual components |
| **Optimization** | Manual trial-and-error | Automatic optimization |
| **Maintenance** | Change prompts everywhere | Update module once |
| **Type Safety** | String manipulation | Typed interfaces |
| **Error Handling** | Parse errors manually | Structured exceptions |

## Real-World Example

Consider building a customer support system:

**Traditional Prompting Approach:**
```python
def handle_customer_query(query):
    # Prompt 1: Classify intent
    classify_prompt = f"""Classify this query: {query}
    Categories: billing, technical, general"""
    intent = llm.complete(classify_prompt)
    
    # Prompt 2: Generate response based on intent
    if "billing" in intent:
        response_prompt = f"""Answer this billing question: {query}
        Be helpful and mention our refund policy."""
    elif "technical" in intent:
        response_prompt = f"""Provide technical support for: {query}
        Include troubleshooting steps."""
    else:
        response_prompt = f"""Answer this question: {query}"""
    
    response = llm.complete(response_prompt)
    return response
```

**LogiLLM Programming Approach:**
```python
from logillm import Module, Predict, ChainOfThought

class CustomerSupport(Module):
    def __init__(self):
        super().__init__()
        # Declare components
        self.classifier = Predict("query -> intent, confidence")
        self.billing = ChainOfThought("query -> solution, next_steps")
        self.technical = ChainOfThought("query -> diagnosis, steps")
        self.general = Predict("query -> response")
    
    async def forward(self, query: str):
        # Classify intent
        intent = await self.classifier(query=query)
        
        # Route to appropriate handler
        if intent.intent == "billing":
            return await self.billing(query=query)
        elif intent.intent == "technical":
            return await self.technical(query=query)
        else:
            return await self.general(query=query)

# Define evaluation metric
def support_metric(prediction, reference):
    return prediction.solution == reference.expected_solution

# Create, optimize, and use
support = CustomerSupport()
optimizer = HybridOptimizer(metric=support_metric)
optimized_support = optimizer.optimize(support, dataset=training_queries)

# Clean usage
response = optimized_support(query="How do I reset my password?")
```

**Benefits of the programming approach:**
1. **Modularity**: Each component can be tested independently
2. **Reusability**: Components can be used in other systems
3. **Optimization**: Entire pipeline can be optimized automatically
4. **Maintainability**: Clear structure and interfaces
5. **Type Safety**: Inputs and outputs are typed
6. **Evolution**: Easy to add new intents or change routing logic

## The Zero-Dependency Philosophy

LogiLLM's programming paradigm extends to its architecture:

```python
# No required dependencies for core functionality
import logillm  # Works immediately

# Optional providers only when needed
# pip install logillm[openai]  # Only if using OpenAI
```

This means:
- **Lightweight**: Core library is pure Python
- **Flexible**: Choose only what you need
- **Testable**: Mock provider for testing without API calls
- **Portable**: Runs anywhere Python runs

## Design Patterns Enabled

The programming paradigm enables software engineering best practices:

### Strategy Pattern
```python
# Different strategies for same task
predictor = Predict(signature)          # Simple prediction
reasoner = ChainOfThought(signature)    # With reasoning
reactor = ReAct(signature)              # With actions
```

### Decorator Pattern
```python
# Enhance modules with additional capabilities
base = Predict("question -> answer")
with_retry = Retry(base, max_retries=3)
with_refine = Refine(with_retry, iterations=2)
with_cache = Cached(with_refine)
```

### Factory Pattern
```python
# Create modules based on configuration
def create_module(task_type: str) -> Module:
    if task_type == "simple":
        return Predict(signature)
    elif task_type == "complex":
        return ChainOfThought(signature)
    elif task_type == "interactive":
        return ReAct(signature)
```

## Summary

LogiLLM's philosophy represents a fundamental shift in how we work with LLMs:

1. **From strings to programs**: Structure over chaos
2. **From manual to automatic**: Optimization built-in
3. **From monolithic to modular**: Composable components
4. **From imperative to declarative**: Say what, not how
5. **From fragile to robust**: Engineering over crafting

This paradigm makes LLM applications:
- More reliable
- Easier to maintain
- Automatically optimizable
- Properly testable
- Production-ready

The result is a framework that brings software engineering discipline to LLM applications while maintaining the flexibility to handle diverse use cases.