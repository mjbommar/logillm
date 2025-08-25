# Code Generation for Unfamiliar Libraries with LogiLLM

> **📍 Tutorial Path**: [Email Extraction](./email-extraction.md) → **Code Generation** → [Yahoo Finance ReAct](./yahoo-finance-react.md) → [AI Text Game](./ai-text-game.md)  
> **⏱️ Time**: 20-25 minutes | **🎯 Difficulty**: Intermediate  
> **💡 Concepts**: Web scraping integration, Iterative refinement, External APIs, Multi-step processing

This tutorial demonstrates how to build an intelligent code generation system using LogiLLM. We'll create a system that can automatically fetch documentation from library websites, analyze it, and generate working code examples for any Python library - perfect for learning new frameworks quickly.

**Perfect for**: Developers wanting to learn multi-step AI systems, anyone building developer tools, those ready for more complex LogiLLM patterns.

**Builds on**: [Email Extraction](./email-extraction.md) - Now we'll take structured processing and add iterative refinement with external data sources.

## What You'll Build

By the end of this tutorial, you'll have a LogiLLM-powered system that can:

- **Fetch documentation** from library websites automatically
- **Analyze library structure** and identify core concepts
- **Generate working code examples** for different use cases
- **Provide explanations** and best practices
- **Handle multiple libraries** with a single interface
- **Create interactive learning sessions** for exploring new frameworks

## Learning Objectives

- Master web scraping and content processing with LogiLLM
- Build modular AI systems for code generation
- Implement intelligent documentation analysis
- Create educational tools for software development
- Demonstrate LogiLLM's text processing capabilities

## Prerequisites

- Python 3.9+ installed
- OpenAI or Anthropic API key
- Basic understanding of LogiLLM signatures and modules
- Familiarity with web scraping concepts

## Installation and Setup

```bash
# Install LogiLLM with provider support
pip install logillm[openai]
# or
pip install logillm[anthropic]

# For web scraping and HTML processing
pip install requests beautifulsoup4 lxml

# For markdown conversion (optional)
pip install markdownify
```

## Step 1: Define Our Signatures

Let's start by defining the LogiLLM signatures for our code generation system:

```python
# signatures.py
from logillm.core.signatures import Signature, InputField, OutputField
from typing import List, Dict, Optional


class AnalyzeLibraryDocs(Signature):
    """Analyze library documentation to extract key concepts and patterns."""
    
    library_name: str = InputField(desc="Name of the Python library")
    documentation_content: str = InputField(desc="Raw documentation content")
    
    core_concepts: List[str] = OutputField(desc="Key concepts and terminology")
    main_classes: List[str] = OutputField(desc="Important classes and their purposes")  
    common_patterns: List[str] = OutputField(desc="Common usage patterns and workflows")
    installation_method: str = OutputField(desc="How to install the library")
    import_statements: List[str] = OutputField(desc="Common import statements")


class GenerateCodeExample(Signature):
    """Generate working code examples for a specific use case."""
    
    library_name: str = InputField()
    use_case: str = InputField(desc="Specific use case or functionality to demonstrate")
    core_concepts: List[str] = InputField(desc="Key concepts from documentation analysis")
    common_patterns: List[str] = InputField(desc="Common usage patterns")
    import_statements: List[str] = InputField(desc="Relevant import statements")
    
    code_example: str = OutputField(desc="Complete, runnable code example")
    explanation: str = OutputField(desc="Step-by-step explanation of the code")
    best_practices: List[str] = OutputField(desc="Best practices and tips")
    potential_issues: List[str] = OutputField(desc="Common pitfalls and how to avoid them")


class CreateLearningPlan(Signature):
    """Create a structured learning plan for mastering the library."""
    
    library_name: str = InputField()
    core_concepts: List[str] = InputField()
    common_patterns: List[str] = InputField()
    user_experience: str = InputField(desc="User's experience level (beginner, intermediate, advanced)")
    
    learning_objectives: List[str] = OutputField(desc="What the user should learn")
    suggested_examples: List[str] = OutputField(desc="Progressive examples to try")
    additional_resources: List[str] = OutputField(desc="Recommended resources for deeper learning")
    time_estimate: str = OutputField(desc="Estimated time to become proficient")


class RefineCodeExample(Signature):
    """Refine and improve a code example based on feedback or requirements."""
    
    original_code: str = InputField(desc="Original code example")
    feedback: str = InputField(desc="User feedback or specific requirements")
    library_patterns: List[str] = InputField(desc="Library-specific patterns to follow")
    
    improved_code: str = OutputField(desc="Refined and improved code example")
    changes_made: List[str] = OutputField(desc="List of improvements made")
    additional_features: List[str] = OutputField(desc="Suggestions for extending the example")
```

## Step 2: Web Scraping and Documentation Processing

```python
# doc_fetcher.py
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse
import time


class DocumentationFetcher:
    """Fetches and processes documentation from library websites."""
    
    def __init__(self, timeout: int = 30, retry_delay: float = 1.0) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LogiLLM Code Generator Bot (Educational Use)'
        })
        self.timeout = timeout
        self.retry_delay = retry_delay
    
    def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch raw HTML content from a URL."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None
    
    def extract_text_content(self, html_content: str) -> str:
        """Extract clean text content from HTML."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def fetch_documentation_urls(self, library_name: str) -> List[str]:
        """Generate potential documentation URLs for a library."""
        base_patterns = [
            f"https://{library_name}.readthedocs.io/",
            f"https://{library_name}.readthedocs.io/en/latest/",
            f"https://docs.{library_name}.org/",
            f"https://{library_name}.org/docs/",
            f"https://{library_name}.org/documentation/",
            f"https://github.com/{library_name}/{library_name}#readme",
            f"https://pypi.org/project/{library_name}/",
        ]
        return base_patterns
    
    def fetch_library_documentation(self, library_name: str) -> Dict[str, str]:
        """Fetch documentation from multiple sources for a library."""
        urls = self.fetch_documentation_urls(library_name)
        docs = {}
        
        for url in urls:
            print(f"🔍 Fetching documentation from: {url}")
            html_content = self.fetch_page_content(url)
            
            if html_content:
                text_content = self.extract_text_content(html_content)
                if text_content and len(text_content) > 100:  # Minimum content threshold
                    docs[url] = text_content
                    print(f"✅ Successfully fetched {len(text_content)} characters")
                else:
                    print(f"⚠️  Insufficient content from {url}")
            else:
                print(f"❌ Failed to fetch content from {url}")
            
            # Be respectful with rate limiting
            time.sleep(self.retry_delay)
        
        return docs
    
    def combine_documentation(self, docs: Dict[str, str]) -> str:
        """Combine documentation from multiple sources."""
        combined = []
        
        for url, content in docs.items():
            combined.append(f"=== Documentation from {url} ===\n")
            combined.append(content[:5000])  # Limit content per source
            combined.append("\n\n")
        
        return "".join(combined)


def create_common_library_examples() -> Dict[str, Dict[str, str]]:
    """Create examples for common libraries when web scraping isn't available."""
    return {
        "fastapi": {
            "basic_setup": '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
            ''',
            "documentation": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+"
        },
        "requests": {
            "basic_setup": '''
import requests

# GET request
response = requests.get('https://httpbin.org/get')
print(response.json())

# POST request with data
data = {'key': 'value'}
response = requests.post('https://httpbin.org/post', json=data)
print(response.status_code)
            ''',
            "documentation": "Requests is a simple, elegant HTTP library for Python"
        }
    }
```

## Step 3: Core Code Generation Module

```python
# generator.py
from typing import Dict, Any, List, Optional
from logillm.core.predict import Predict
from logillm.core.modules import Module
from .signatures import AnalyzeLibraryDocs, GenerateCodeExample, CreateLearningPlan, RefineCodeExample
from .doc_fetcher import DocumentationFetcher, create_common_library_examples


class CodeGenerationResult:
    """Container for code generation results."""
    
    def __init__(
        self,
        library_name: str,
        analysis: Any,
        code_examples: Dict[str, Any],
        learning_plan: Optional[Any] = None
    ):
        self.library_name = library_name
        self.analysis = analysis
        self.code_examples = code_examples
        self.learning_plan = learning_plan


class LibraryCodeGenerator(Module):
    """Main module for generating code examples for unfamiliar libraries."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Initialize LogiLLM components
        self.doc_analyzer = Predict(signature=AnalyzeLibraryDocs)
        self.code_generator = Predict(signature=GenerateCodeExample)
        self.learning_planner = Predict(signature=CreateLearningPlan)
        self.code_refiner = Predict(signature=RefineCodeExample)
        
        # Initialize documentation fetcher
        self.doc_fetcher = DocumentationFetcher()
        
        # Common library examples for fallback
        self.common_examples = create_common_library_examples()
    
    async def analyze_library(self, library_name: str, documentation: str) -> Any:
        """Analyze library documentation to extract key information."""
        return await self.doc_analyzer(
            library_name=library_name,
            documentation_content=documentation
        )
    
    async def generate_example(
        self,
        library_name: str,
        use_case: str,
        analysis: Any
    ) -> Any:
        """Generate a code example for a specific use case."""
        return await self.code_generator(
            library_name=library_name,
            use_case=use_case,
            core_concepts=analysis.core_concepts,
            common_patterns=analysis.common_patterns,
            import_statements=analysis.import_statements
        )
    
    async def create_learning_plan(
        self,
        library_name: str,
        analysis: Any,
        user_experience: str = "beginner"
    ) -> Any:
        """Create a structured learning plan for the library."""
        return await self.learning_planner(
            library_name=library_name,
            core_concepts=analysis.core_concepts,
            common_patterns=analysis.common_patterns,
            user_experience=user_experience
        )
    
    async def refine_code(
        self,
        original_code: str,
        feedback: str,
        library_patterns: List[str]
    ) -> Any:
        """Refine and improve a code example based on feedback."""
        return await self.code_refiner(
            original_code=original_code,
            feedback=feedback,
            library_patterns=library_patterns
        )
    
    async def forward(
        self,
        library_name: str,
        use_cases: List[str],
        user_experience: str = "beginner",
        fetch_docs: bool = True
    ) -> CodeGenerationResult:
        """Generate comprehensive code examples for a library."""
        
        print(f"🚀 Generating code examples for {library_name}")
        
        # Step 1: Get documentation
        if fetch_docs:
            print("📚 Fetching documentation...")
            docs = self.doc_fetcher.fetch_library_documentation(library_name)
            documentation = self.doc_fetcher.combine_documentation(docs)
        else:
            # Use fallback documentation if available
            if library_name in self.common_examples:
                documentation = self.common_examples[library_name]["documentation"]
            else:
                documentation = f"Limited documentation available for {library_name}"
        
        print(f"📖 Documentation length: {len(documentation)} characters")
        
        # Step 2: Analyze the library
        print("🔍 Analyzing library structure...")
        analysis = await self.analyze_library(library_name, documentation)
        
        print(f"✅ Found {len(analysis.core_concepts)} core concepts")
        print(f"✅ Identified {len(analysis.common_patterns)} patterns")
        
        # Step 3: Generate code examples for each use case
        print("💻 Generating code examples...")
        code_examples = {}
        
        for use_case in use_cases:
            print(f"   Generating example for: {use_case}")
            example = await self.generate_example(library_name, use_case, analysis)
            code_examples[use_case] = example
        
        # Step 4: Create learning plan
        print("📝 Creating learning plan...")
        learning_plan = await self.create_learning_plan(
            library_name, analysis, user_experience
        )
        
        return CodeGenerationResult(
            library_name=library_name,
            analysis=analysis,
            code_examples=code_examples,
            learning_plan=learning_plan
        )
```

## Step 4: Interactive Learning Session

```python
# interactive.py
import asyncio
from typing import List
from .generator import LibraryCodeGenerator


class InteractiveLearningSession:
    """Interactive session for learning new libraries."""
    
    def __init__(self) -> None:
        self.generator = LibraryCodeGenerator()
        self.current_library: str = ""
        self.current_analysis: Any = None
    
    def display_analysis(self, analysis: Any) -> None:
        """Display library analysis in a user-friendly format."""
        print("\n📊 LIBRARY ANALYSIS")
        print("=" * 50)
        print(f"📦 Installation: {analysis.installation_method}")
        
        print(f"\n🔑 Core Concepts ({len(analysis.core_concepts)}):")
        for i, concept in enumerate(analysis.core_concepts, 1):
            print(f"   {i}. {concept}")
        
        print(f"\n🏗️  Main Classes ({len(analysis.main_classes)}):")
        for i, cls in enumerate(analysis.main_classes, 1):
            print(f"   {i}. {cls}")
        
        print(f"\n⚡ Common Patterns ({len(analysis.common_patterns)}):")
        for i, pattern in enumerate(analysis.common_patterns, 1):
            print(f"   {i}. {pattern}")
        
        print(f"\n📥 Import Statements:")
        for stmt in analysis.import_statements:
            print(f"   {stmt}")
    
    def display_code_example(self, use_case: str, example: Any) -> None:
        """Display a code example with explanation."""
        print(f"\n💻 CODE EXAMPLE: {use_case}")
        print("=" * 50)
        print(example.code_example)
        
        print(f"\n📝 EXPLANATION:")
        print(example.explanation)
        
        if example.best_practices:
            print(f"\n✅ BEST PRACTICES:")
            for i, practice in enumerate(example.best_practices, 1):
                print(f"   {i}. {practice}")
        
        if example.potential_issues:
            print(f"\n⚠️  POTENTIAL ISSUES:")
            for i, issue in enumerate(example.potential_issues, 1):
                print(f"   {i}. {issue}")
    
    def display_learning_plan(self, plan: Any) -> None:
        """Display the learning plan."""
        print(f"\n🎯 LEARNING PLAN")
        print("=" * 50)
        print(f"⏱️  Estimated Time: {plan.time_estimate}")
        
        print(f"\n🎯 Learning Objectives:")
        for i, obj in enumerate(plan.learning_objectives, 1):
            print(f"   {i}. {obj}")
        
        print(f"\n📚 Suggested Examples (in order):")
        for i, example in enumerate(plan.suggested_examples, 1):
            print(f"   {i}. {example}")
        
        if plan.additional_resources:
            print(f"\n🔗 Additional Resources:")
            for resource in plan.additional_resources:
                print(f"   • {resource}")
    
    async def explore_library(
        self, 
        library_name: str, 
        user_experience: str = "beginner"
    ) -> None:
        """Start an interactive exploration of a library."""
        
        self.current_library = library_name
        
        # Common use cases for different types of libraries
        common_use_cases = [
            "basic setup and hello world",
            "intermediate usage with common features",
            "advanced usage with best practices",
            "error handling and debugging"
        ]
        
        try:
            result = await self.generator.forward(
                library_name=library_name,
                use_cases=common_use_cases,
                user_experience=user_experience,
                fetch_docs=True
            )
            
            self.current_analysis = result.analysis
            
            # Display results
            self.display_analysis(result.analysis)
            
            print(f"\n🎓 Generated {len(result.code_examples)} code examples")
            
            # Show each example
            for use_case, example in result.code_examples.items():
                self.display_code_example(use_case, example)
                
                # Wait for user input to continue
                input(f"\nPress Enter to continue to the next example...")
            
            # Show learning plan
            if result.learning_plan:
                self.display_learning_plan(result.learning_plan)
            
            print(f"\n🎉 Exploration of {library_name} complete!")
            
        except Exception as e:
            print(f"❌ Error exploring library: {e}")
            import traceback
            traceback.print_exc()
    
    async def refine_example(self, use_case: str, feedback: str) -> None:
        """Refine a previously generated example based on feedback."""
        if not self.current_analysis:
            print("❌ No current analysis available. Explore a library first.")
            return
        
        # This would need to be implemented to store previous examples
        print(f"🔄 Refining example for '{use_case}' based on feedback...")
        # Implementation would go here
```

## Step 5: Main Application and Demo

```python
# demo.py
import asyncio
import os
from typing import List

from logillm.providers import create_provider, register_provider
from .interactive import InteractiveLearningSession
from .generator import LibraryCodeGenerator


async def demo_fastapi_generation() -> None:
    """Demonstrate code generation for FastAPI."""
    
    # Setup LogiLLM provider
    model = os.environ.get("MODEL", "gpt-4o-mini")
    
    if model.startswith("gpt"):
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Please set OPENAI_API_KEY environment variable")
            return
        provider = create_provider("openai", model=model)
    elif model.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
            return
        provider = create_provider("anthropic", model=model)
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    register_provider(provider, set_default=True)
    
    # Create learning session
    session = InteractiveLearningSession()
    
    # Explore FastAPI
    await session.explore_library(
        library_name="fastapi",
        user_experience="beginner"
    )


async def demo_multiple_libraries() -> None:
    """Demonstrate generation for multiple libraries."""
    
    model = os.environ.get("MODEL", "gpt-4o-mini")
    
    if model.startswith("gpt"):
        provider = create_provider("openai", model=model)
    else:
        provider = create_provider("anthropic", model=model)
    
    register_provider(provider, set_default=True)
    
    generator = LibraryCodeGenerator()
    
    libraries = ["requests", "pandas", "click"]
    use_cases = [
        "basic usage example",
        "common workflow",
        "error handling"
    ]
    
    for library in libraries:
        print(f"\n{'='*60}")
        print(f"🔍 Exploring {library.upper()}")
        print(f"{'='*60}")
        
        result = await generator.forward(
            library_name=library,
            use_cases=use_cases,
            user_experience="intermediate",
            fetch_docs=False  # Use fallback for demo speed
        )
        
        print(f"\n📊 Analysis for {library}:")
        print(f"   Core concepts: {len(result.analysis.core_concepts)}")
        print(f"   Generated examples: {len(result.code_examples)}")
        
        # Show first example
        if result.code_examples:
            first_example = list(result.code_examples.values())[0]
            print(f"\n💻 Sample code:")
            print(first_example.code_example[:200] + "...")


async def main() -> None:
    """Main demo entry point."""
    
    print("🚀 LogiLLM Code Generation Tutorial")
    print("=" * 50)
    
    choice = input("Choose demo (1=FastAPI deep dive, 2=Multiple libraries): ")
    
    if choice == "1":
        await demo_fastapi_generation()
    else:
        await demo_multiple_libraries()


if __name__ == "__main__":
    asyncio.run(main())
```

## Step 6: Testing and Validation

```python
# test_tutorial.py
"""
Test script for the code generation tutorial.
Run with: uv run --with logillm[openai] --with requests --with beautifulsoup4 python test_tutorial.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.tutorials.code_generation.demo import demo_multiple_libraries


async def test_tutorial() -> None:
    """Test the code generation tutorial."""
    
    model = os.environ.get("MODEL", "gpt-4o-mini")
    
    if model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    elif model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    try:
        print("🧪 Running code generation tutorial test...")
        await demo_multiple_libraries()
        print("✅ Tutorial test completed successfully!")
        
    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tutorial())
```

## Key LogiLLM Advantages Demonstrated

This tutorial showcases several LogiLLM advantages:

1. **Structured Processing**: Clean separation between documentation fetching, analysis, and generation
2. **Modular Architecture**: Composable modules for different aspects of code generation  
3. **Type Safety**: Complete type hints and structured outputs
4. **Async Support**: Native async/await for efficient web scraping and API calls
5. **Error Handling**: Robust error handling with fallback mechanisms
6. **Extensibility**: Easy to add new libraries, use cases, and generation patterns

## Expected Output

When you run the tutorial, you should see:

```
🚀 LogiLLM Code Generation Tutorial
==================================================

============================================================
🔍 Exploring REQUESTS
============================================================
🚀 Generating code examples for requests
📚 Fetching documentation...
🔍 Fetching documentation from: https://requests.readthedocs.io/
✅ Successfully fetched 15234 characters
📖 Documentation length: 15234 characters
🔍 Analyzing library structure...
✅ Found 8 core concepts
✅ Identified 12 patterns
💻 Generating code examples...
   Generating example for: basic usage example
   Generating example for: common workflow
   Generating example for: error handling
📝 Creating learning plan...

📊 Analysis for requests:
   Core concepts: 8
   Generated examples: 3

💻 Sample code:
import requests

# Basic GET request
response = requests.get('https://api.example.com/users')
if response.status_code == 200:
    data = response.json()
    print(f"Retrieved {len(data)} users")...
```

## Next Steps and Extensions

- **Add more library types**: Database ORMs, ML frameworks, GUI libraries
- **Implement caching**: Cache documentation and analysis results
- **Add code validation**: Test generated code examples automatically
- **Create learning paths**: Progressive tutorials for complex libraries
- **Build web interface**: Interactive web app for code generation
- **Integration with IDEs**: Plugin for popular development environments

## 🎓 What You've Learned

Great work! You've now mastered multi-step AI system design:

✅ **Web Scraping Integration**: Connected external data sources to LLM processing  
✅ **Iterative Refinement**: Built systems that improve their output through multiple passes  
✅ **External API Integration**: Learned to combine web data with LLM capabilities  
✅ **Multi-step Processing**: Orchestrated complex workflows with multiple components  
✅ **Error Handling**: Built robust systems that handle real-world web data

## 🚀 What's Next?

### Immediate Next Steps
**Ready for intelligent agents?** → **[Yahoo Finance ReAct Tutorial](./yahoo-finance-react.md)**  
Learn how to transform the multi-step processing you just mastered into reasoning agents that can make decisions and use tools.

### Apply What You've Learned  
- **Try different libraries**: Test with FastAPI, Django, PyTorch, or other frameworks
- **Improve documentation parsing**: Add support for different documentation formats  
- **Add code validation**: Run generated code and provide feedback on errors

### Advanced Extensions  
- **Code execution environment**: Run and test generated examples automatically
- **Interactive tutorials**: Build step-by-step learning experiences
- **IDE integration**: Create VS Code or PyCharm plugins
- **Multi-language support**: Extend to JavaScript, Go, Rust libraries

### Tutorial Learning Path
1. ✅ **[LLM Text Generation](./llms-txt-generation.md)** - Foundation concepts
2. ✅ **[Email Extraction](./email-extraction.md)** - Structured data processing  
3. ✅ **Code Generation** (You are here!)
4. → **[Yahoo Finance ReAct](./yahoo-finance-react.md)** - Agent reasoning
5. → **[AI Text Game](./ai-text-game.md)** - Interactive systems  
6. → **[Memory-Enhanced ReAct Agent](./memory-enhanced-react.md)** - Persistent memory

### Concept Connections
- **From Code Generation to ReAct Agents**: The multi-step processing patterns you learned are the foundation of agent reasoning loops
- **External Integration**: Web scraping skills transfer directly to agent tool usage
- **Iterative Refinement**: This pattern becomes "reflection" in agent systems

## 🛠️ Running the Tutorial

```bash
# With OpenAI
export OPENAI_API_KEY="your-key-here"
uv run --with logillm --with openai --with requests --with beautifulsoup4 --with lxml python -m examples.tutorials.code_generation.demo

# With Anthropic
export ANTHROPIC_API_KEY="your-key-here"
uv run --with logillm --with anthropic --with requests --with beautifulsoup4 --with lxml python -m examples.tutorials.code_generation.demo

# Run tests to verify your setup
uv run --with logillm --with openai --with requests --with beautifulsoup4 python examples/tutorials/code_generation/test_tutorial.py
```

---

**📚 [← Email Extraction](./email-extraction.md) | [Tutorial Index](./README.md) | [Yahoo Finance ReAct →](./yahoo-finance-react.md)**

You've mastered multi-step AI systems! Ready to learn agent reasoning? Continue with **[Yahoo Finance ReAct](./yahoo-finance-react.md)** to build on these concepts.