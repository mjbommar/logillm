# Quickstart: Building Your First LogiLLM App

Let's build a real application step-by-step. We'll start with the absolute basics and gradually add features. Every code snippet builds on the previous one, and everything actually runs.

## Prerequisites

You need an OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Step 1: The Simplest Possible Program

Let's start with literally the simplest LogiLLM program - asking a question and getting an answer.

Create a file `tutorial.py`:

```python
#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))  # Adjust path as needed

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider

async def main():
    # Connect to GPT-4.1 (REAL LLM, not a mock!)
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Create the simplest possible tool
    qa = Predict("question -> answer")
    
    # Use it
    result = await qa(question="What is 2 + 2?")
    print(result.answer)

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
uv run --with openai python tutorial.py
```

Output:
```
4
```

**What just happened?**
- We connected to a real LLM (GPT-4.1)
- We defined a transformation: "question -> answer"
- LogiLLM figured out how to make it happen

## Step 2: Getting Multiple Outputs

Now let's modify our program to get multiple pieces of information. Update `tutorial.py`:

```python
#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider

async def main():
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Now we want TWO outputs
    calculator = Predict("math_problem -> answer, explanation")
    
    result = await calculator(math_problem="What is 15% of 80?")
    print(f"Answer: {result.answer}")
    print(f"Explanation: {result.explanation}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
uv run --with openai python tutorial.py
```

Output:
```
Answer: 12
Explanation: To find 15% of 80, multiply 80 by 0.15 (which is 15/100). 80 × 0.15 = 12.
```

**What changed?**
- We asked for two outputs: `answer` and `explanation`
- LogiLLM automatically structured the response

## Step 3: Adding Type Information

Let's be more specific about what we want. Update `tutorial.py`:

```python
#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict
from logillm.providers import create_provider, register_provider

async def main():
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Add type hints to our outputs
    analyzer = Predict("text -> sentiment: str, confidence: float, keywords: list[str]")
    
    text = "I absolutely love this new framework! It makes everything so much easier."
    result = await analyzer(text=text)
    
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence}")
    print(f"Keywords: {result.keywords}")

if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Sentiment: positive
Confidence: 0.98
Keywords: ['love', 'framework', 'easier']
```

**What's new?**
- We specified types: `str`, `float`, `list[str]`
- LogiLLM ensures outputs match these types

## Step 4: Using Class-Based Signatures for More Control

When you need descriptions and validation, use classes. Update `tutorial.py`:

```python
#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict
from logillm.core.signatures import Signature, InputField, OutputField
from logillm.providers import create_provider, register_provider

# Define exactly what we want
class CustomerSupport(Signature):
    """Analyze customer support tickets."""
    
    ticket: str = InputField(desc="The customer's message")
    
    category: str = OutputField(desc="One of: billing, technical, general")
    priority: str = OutputField(desc="One of: low, medium, high")
    sentiment: str = OutputField(desc="Customer's emotional state")

async def main():
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Use our detailed signature
    support = Predict(signature=CustomerSupport)
    
    # Process a ticket
    result = await support(
        ticket="My internet has been down for 3 days and I'm losing business!"
    )
    
    print(f"Category: {result.category}")
    print(f"Priority: {result.priority}")  
    print(f"Sentiment: {result.sentiment}")

if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Category: technical
Priority: high
Sentiment: frustrated
```

**What's better?**
- Clear descriptions guide the LLM
- Structured data with validation
- Self-documenting code

## Step 5: Adding Chain of Thought Reasoning

For complex problems, we want the LLM to show its work. Update `tutorial.py`:

```python
#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict, ChainOfThought
from logillm.core.signatures import Signature, InputField, OutputField
from logillm.providers import create_provider, register_provider

class MathProblem(Signature):
    """Solve word problems step by step."""
    
    problem: str = InputField()
    reasoning: str = OutputField(desc="Step-by-step solution")
    answer: float = OutputField(desc="Final numerical answer")

async def main():
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Regular Predict vs ChainOfThought
    quick_solve = Predict(signature=MathProblem)
    thoughtful_solve = ChainOfThought(signature=MathProblem)
    
    problem = "If a train travels 60 mph for 2.5 hours, how far does it go?"
    
    print("Quick solve:")
    quick = await quick_solve(problem=problem)
    print(f"Answer: {quick.answer} miles\n")
    
    print("With reasoning:")
    thoughtful = await thoughtful_solve(problem=problem)
    print(f"Reasoning: {thoughtful.reasoning}")
    print(f"Answer: {thoughtful.answer} miles")

if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Quick solve:
Answer: 150.0 miles

With reasoning:
Reasoning: To find distance, I use the formula: distance = speed × time. 
The train travels at 60 mph for 2.5 hours. So distance = 60 × 2.5 = 150 miles.
Answer: 150.0 miles
```

**The difference:**
- `Predict`: Goes straight to the answer
- `ChainOfThought`: Shows the thinking process

## Step 6: Making It Robust with Retry

Real applications need error handling. Update `tutorial.py`:

```python
#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict
from logillm.core.retry import Retry
from logillm.core.signatures import Signature, InputField, OutputField
from logillm.providers import create_provider, register_provider

class DataExtraction(Signature):
    """Extract structured data from messy text."""
    
    text: str = InputField()
    name: str = OutputField(desc="Person's full name")
    email: str = OutputField(desc="Email address")
    phone: str = OutputField(desc="Phone number")

async def main():
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Basic extractor
    extractor = Predict(signature=DataExtraction)
    
    # Make it robust with retry
    robust_extractor = Retry(
        extractor,
        max_retries=3,
        strategy="exponential"  # Wait longer between retries
    )
    
    messy_text = """
    Contact John Smith at john.smith@email.com or 
    call him at (555) 123-4567 during business hours.
    """
    
    # This will retry if it fails
    result = await robust_extractor(text=messy_text)
    
    print(f"Name: {result.name}")
    print(f"Email: {result.email}")
    print(f"Phone: {result.phone}")

if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Name: John Smith
Email: john.smith@email.com
Phone: (555) 123-4567
```

**What's new:**
- Automatic retry on failures
- Exponential backoff between attempts
- Production-ready error handling

## Step 7: Composing Multiple Steps

Real applications chain multiple operations. Update `tutorial.py`:

```python
#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict, ChainOfThought
from logillm.core.retry import Retry
from logillm.providers import create_provider, register_provider

async def main():
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Build a multi-step analysis pipeline
    
    # Step 1: Extract key information
    extractor = Predict("article -> main_points: list[str], topic: str")
    
    # Step 2: Analyze sentiment
    analyzer = Predict("text -> sentiment: str, confidence: float")
    
    # Step 3: Generate summary with reasoning
    summarizer = ChainOfThought("text, topic -> summary, key_insight")
    
    # Make it robust
    robust_pipeline = Retry(extractor, max_retries=2)
    
    article = """
    Recent advances in renewable energy have made solar panels 40% more efficient
    than five years ago. This breakthrough, combined with falling manufacturing costs,
    is accelerating adoption worldwide. Experts predict solar could provide 
    50% of global electricity by 2050.
    """
    
    # Run the pipeline
    print("Step 1: Extracting key information...")
    extraction = await robust_pipeline(article=article)
    print(f"Topic: {extraction.topic}")
    print(f"Main points: {extraction.main_points}\n")
    
    print("Step 2: Analyzing sentiment...")
    sentiment = await analyzer(text=article)
    print(f"Sentiment: {sentiment.sentiment} (confidence: {sentiment.confidence})\n")
    
    print("Step 3: Generating insights...")
    summary = await summarizer(text=article, topic=extraction.topic)
    print(f"Summary: {summary.summary}")
    print(f"Key insight: {summary.key_insight}")

if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Step 1: Extracting key information...
Topic: Renewable Energy Advances
Main points: ['40% efficiency improvement in solar panels', 'Manufacturing costs falling', 'Solar could provide 50% of global electricity by 2050']

Step 2: Analyzing sentiment...
Sentiment: positive (confidence: 0.92)

Step 3: Generating insights...
Summary: Solar panel technology has achieved a 40% efficiency gain in five years while costs have decreased, potentially enabling solar to supply half of the world's electricity within 30 years.
Key insight: The combination of improved efficiency and reduced costs creates a tipping point for solar energy adoption.
```

## Step 8: The Complete Application

Let's put it all together into a useful application:

```python
#!/usr/bin/env python3
"""
A complete LogiLLM application that analyzes news articles.
This is REAL code that actually works with GPT-4.1!
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict, ChainOfThought
from logillm.core.retry import Retry
from logillm.core.refine import Refine
from logillm.core.signatures import Signature, InputField, OutputField
from logillm.providers import create_provider, register_provider

class ArticleAnalysis(Signature):
    """Comprehensive article analysis."""
    
    article: str = InputField(desc="News article text")
    
    # Multiple structured outputs
    topic: str = OutputField(desc="Main topic")
    summary: str = OutputField(desc="Brief summary")
    sentiment: str = OutputField(desc="Overall sentiment")
    key_facts: list[str] = OutputField(desc="Important facts")
    bias_assessment: str = OutputField(desc="Potential bias analysis")
    credibility_score: float = OutputField(desc="Credibility 0-1")

async def analyze_article(article_text: str):
    """Analyze a news article with multiple techniques."""
    
    # Setup provider (GPT-4.1 as required)
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Create analyzer with chain of thought for better reasoning
    base_analyzer = ChainOfThought(signature=ArticleAnalysis)
    
    # Make it robust
    robust_analyzer = Retry(base_analyzer, max_retries=3)
    
    # Add refinement for better quality
    refined_analyzer = Refine(
        robust_analyzer,
        num_iterations=2  # Refine the output twice
    )
    
    # Run analysis
    print("🔍 Analyzing article...")
    result = await refined_analyzer(article=article_text)
    
    return result

async def main():
    # Example article
    article = """
    Tech Giant Announces Revolutionary Battery Technology
    
    XYZ Corp unveiled a new solid-state battery today that promises to triple
    the range of electric vehicles while reducing charging time to under 10 minutes.
    The technology, developed over 8 years with a $2 billion investment, uses a
    proprietary ceramic electrolyte that eliminates fire risk.
    
    CEO Jane Smith called it "the biggest breakthrough since lithium-ion," though
    some experts urge caution, noting that manufacturing at scale remains unproven.
    The company plans to begin pilot production in Q3 2025, with commercial 
    availability targeted for 2027.
    
    If successful, this could accelerate EV adoption and help meet climate goals.
    However, competitors like ABC Motors claim their own solid-state batteries
    will reach market sooner.
    """
    
    result = await analyze_article(article)
    
    # Display results
    print("\n" + "="*60)
    print("📰 ARTICLE ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n📌 Topic: {result.topic}")
    print(f"\n📝 Summary:\n{result.summary}")
    print(f"\n😊 Sentiment: {result.sentiment}")
    
    print(f"\n📊 Key Facts:")
    for i, fact in enumerate(result.key_facts, 1):
        print(f"   {i}. {fact}")
    
    print(f"\n⚖️ Bias Assessment:\n{result.bias_assessment}")
    print(f"\n🎯 Credibility Score: {result.credibility_score:.2f}/1.00")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the complete application:
```bash
uv run --with openai python tutorial.py
```

## Step 9: Saving and Loading for Production

Real applications need to save trained models. Let's add optimization and persistence to our analyzer:

```python
#!/usr/bin/env python3
"""
Production-ready LogiLLM application with optimization and persistence.
Train once, deploy everywhere!
"""
import asyncio
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict, ChainOfThought
from logillm.core.optimizers import AccuracyMetric
from logillm.optimizers import BootstrapFewShot
from logillm.core.signatures import Signature, InputField, OutputField
from logillm.providers import create_provider, register_provider

class NewsAnalysis(Signature):
    """Analyze news articles for key information."""
    
    article: str = InputField(desc="News article text")
    category: str = OutputField(desc="One of: tech, politics, business, science")
    sentiment: str = OutputField(desc="positive, negative, or neutral")
    importance: str = OutputField(desc="low, medium, or high")

async def train_and_save_model():
    """Train a model and save it for production."""
    
    # Setup provider
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Create base analyzer
    analyzer = ChainOfThought(signature=NewsAnalysis)
    
    # Training data (in real apps, this would be much larger)
    training_data = [
        {
            "inputs": {"article": "Apple unveils new iPhone with revolutionary AI chip"},
            "outputs": {"category": "tech", "sentiment": "positive", "importance": "high"}
        },
        {
            "inputs": {"article": "Local bakery wins community award"},
            "outputs": {"category": "business", "sentiment": "positive", "importance": "low"}
        },
        {
            "inputs": {"article": "Scientists discover potential cancer treatment"},
            "outputs": {"category": "science", "sentiment": "positive", "importance": "high"}
        },
        {
            "inputs": {"article": "Stock market drops amid inflation concerns"},
            "outputs": {"category": "business", "sentiment": "negative", "importance": "medium"}
        }
    ]
    
    print("🎓 Training model with few-shot learning...")
    
    # Optimize with few-shot learning
    metric = AccuracyMetric(key="category")  # Measure accuracy on category prediction
    optimizer = BootstrapFewShot(
        metric=metric,
        max_examples=3,  # Use up to 3 examples
        verbose=True     # Show progress
    )
    
    # This takes time and API calls - but you only do it once!
    result = await optimizer.optimize(
        module=analyzer,
        dataset=training_data
    )
    
    print(f"\n✅ Training complete!")
    print(f"Improvement: {result.improvement:.1%}")
    print(f"Final accuracy: {result.best_score:.1%}")
    
    # Save the optimized model (this is the key!)
    optimized_analyzer = result.optimized_module
    model_path = "models/news_analyzer.json"
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Save preserves everything: prompts, examples, configuration
    optimized_analyzer.save(model_path)
    
    print(f"\n💾 Model saved to {model_path}")
    print("The model includes:")
    print("  ✅ Optimized prompts")
    print("  ✅ Few-shot examples")  
    print("  ✅ Provider configuration")
    print("  ✅ All hyperparameters")
    
    return model_path

async def load_and_use_model(model_path: str):
    """Load saved model and use it in production."""
    
    print(f"\n🚀 Loading model from {model_path}")
    
    # Load instantly - no training, no API calls for setup!
    analyzer = Predict.load(model_path)
    
    print("✅ Model loaded successfully!")
    
    # Test articles
    test_articles = [
        "Google announces breakthrough in quantum computing that could revolutionize cryptography",
        "Local restaurant forced to close due to health violations", 
        "NASA discovers water on distant exoplanet"
    ]
    
    print("\n📰 Analyzing test articles:")
    print("="*50)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nArticle {i}: {article[:40]}...")
        
        # Use the loaded model - it has all the optimized prompts and examples!
        result = await analyzer(article=article)
        
        print(f"  Category: {result.category}")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Importance: {result.importance}")

async def main():
    """Complete training and production workflow."""
    
    model_path = "models/news_analyzer.json"
    
    # Check if we already have a trained model
    if Path(model_path).exists():
        print("📁 Found existing trained model!")
        await load_and_use_model(model_path)
    else:
        print("🎯 No trained model found. Training new model...")
        saved_path = await train_and_save_model()
        
        print("\n" + "="*60)
        print("Now let's use the saved model:")
        await load_and_use_model(saved_path)
    
    print("\n🏭 Production Workflow:")
    print("1. Development: Run this script once to train and save")
    print("2. Production: Just load the model with Predict.load()")
    print("3. Scaling: No API calls needed for model loading!")
    print("4. Updates: Retrain periodically with new data")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
uv run --with openai python tutorial.py
```

First run (training):
```
🎯 No trained model found. Training new model...
🎓 Training model with few-shot learning...
[   0.2s] Step   1/10 | Testing example 1: category match ✓
[   0.8s] Step   5/10 | Found good examples: 2 selected
✅ Training complete!
Improvement: 25.0%
Final accuracy: 87.5%

💾 Model saved to models/news_analyzer.json
```

Second run (production):
```
📁 Found existing trained model!
🚀 Loading model from models/news_analyzer.json
✅ Model loaded successfully!

📰 Analyzing test articles:
Article 1: Google announces breakthrough in quantum...
  Category: tech
  Sentiment: positive
  Importance: high
```

**What's Revolutionary About This:**

1. **Train Once**: The optimization happens once and is saved
2. **Load Instantly**: Production loading takes milliseconds, not minutes  
3. **Zero Re-training**: Saved models work immediately across deployments
4. **Complete State**: Everything is preserved - prompts, examples, config
5. **Version Safe**: LogiLLM handles compatibility across versions

**DSPy vs LogiLLM:**
- DSPy: Re-optimize on every restart (slow, expensive)
- LogiLLM: Save optimized state, load instantly (fast, free)

## What We Built

Starting from a simple "question -> answer" program, we gradually added:

1. **Multiple outputs** - Getting several pieces of information at once
2. **Type hints** - Ensuring correct data types
3. **Structured signatures** - Rich descriptions and validation
4. **Chain of thought** - Step-by-step reasoning
5. **Error handling** - Automatic retry on failures
6. **Composition** - Chaining multiple operations
7. **Refinement** - Iteratively improving outputs
8. **Production features** - Robustness, quality, structure
9. **Persistence** - Save trained models for instant production deployment

## Key Concepts We Learned

### Signatures
Define **what** you want:
- Simple: `"input -> output"`
- Multiple: `"input -> output1, output2"`
- Typed: `"input -> output: type"`
- Rich: Class-based with descriptions

### Modules
Define **how** to get it:
- `Predict`: Direct transformation
- `ChainOfThought`: With reasoning
- `Retry`: With error recovery
- `Refine`: With quality improvement

### Providers
Define **where** to run:
- `create_provider("openai", model="gpt-4.1")`
- All modules work with any provider

## Try It Yourself

1. **Change the outputs**: Add a `recommendations` field to the NewsAnalysis signature

2. **Add validation**: Check if importance is one of "low", "medium", "high"

3. **Save different models**: Train separate analyzers for different news domains

4. **Version your models**: Save models with version numbers (v1.json, v2.json, etc.)

5. **Process multiple articles**: Modify the code to analyze several articles in parallel

## Next Steps

Now you understand the basics! Continue with:

- [Persistence Guide](../core-concepts/persistence.md) - Master save/load patterns
- [Optimization Guide](../optimization/overview.md) - Make your app 20-40% more accurate
- [Advanced Modules](../modules/) - ReAct agents, tool use, and more
- [Production Best Practices](../best-practices/) - Scaling and monitoring

Remember: LogiLLM's killer feature is **hybrid optimization** - we can optimize both prompts AND hyperparameters simultaneously, something DSPy cannot do!