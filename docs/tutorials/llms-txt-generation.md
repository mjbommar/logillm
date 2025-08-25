# Generating llms.txt Documentation with LogiLLM

> **📍 Tutorial Path**: Beginner → **LLM Text Generation** → [Email Extraction](./email-extraction.md) → [Code Generation](./code-generation.md)  
> **⏱️ Time**: 10-15 minutes | **🎯 Difficulty**: Beginner  
> **💡 Concepts**: Signatures, Predict modules, Provider setup, Repository analysis

This tutorial demonstrates how to build an intelligent repository documentation generator using LogiLLM. We'll create a system that automatically analyzes GitHub repositories and generates comprehensive `llms.txt` files - a standard for providing LLM-friendly project documentation.

**Perfect for**: First-time LogiLLM users, developers coming from other frameworks, anyone wanting to understand LogiLLM basics.

## What You'll Build

By the end of this tutorial, you'll have a LogiLLM-powered system that can:

- **Analyze repository structure** and identify key components
- **Extract project information** from README files and code
- **Generate structured documentation** following the llms.txt standard
- **Create comprehensive overviews** including purpose, architecture, and usage
- **Save and reuse optimized models** for consistent documentation quality

## What is llms.txt?

`llms.txt` is a proposed standard for providing structured, LLM-friendly documentation about projects. It typically includes:

- Project overview and purpose
- Key concepts and terminology  
- Architecture and structure
- Usage examples
- Important files and directories

## Prerequisites

- Python 3.9+ installed
- OpenAI or Anthropic API key
- Basic understanding of LogiLLM signatures and modules
- GitHub API access (optional, for live repository analysis)

## Installation and Setup

```bash
# Install LogiLLM with OpenAI support
pip install logillm[openai]

# Or with Anthropic support  
pip install logillm[anthropic]

# For GitHub API access
pip install requests
```

## Step 1: Define Our Signatures

First, let's define LogiLLM signatures for different aspects of documentation generation:

```python
# signatures.py
from logillm.core.signatures import Signature, InputField, OutputField
from typing import List

class AnalyzeRepository(Signature):
    """Analyze a repository structure and identify key components."""
    
    repo_url: str = InputField(desc="GitHub repository URL")
    file_tree: str = InputField(desc="Repository file structure as text")
    readme_content: str = InputField(desc="README.md content")
    
    project_purpose: str = OutputField(desc="Main purpose and goals of the project")
    key_concepts: List[str] = OutputField(desc="Important concepts and terminology")
    architecture_overview: str = OutputField(desc="High-level architecture description")

class AnalyzeCodeStructure(Signature):
    """Analyze code structure to identify important directories and files."""
    
    file_tree: str = InputField(desc="Repository file structure")
    package_files: str = InputField(desc="Key package and configuration files")
    
    important_directories: List[str] = OutputField(desc="Key directories and their purposes")
    entry_points: List[str] = OutputField(desc="Main entry points and important files")
    development_info: str = OutputField(desc="Development setup and workflow information")

class GenerateUsageExamples(Signature):
    """Generate practical usage examples based on project information."""
    
    project_purpose: str = InputField()
    key_concepts: List[str] = InputField()
    architecture_overview: str = InputField()
    
    usage_examples: str = OutputField(desc="Practical usage examples and code snippets")

class GenerateLLMsTxt(Signature):
    """Generate a comprehensive llms.txt file from analyzed repository information."""
    
    project_purpose: str = InputField()
    key_concepts: List[str] = InputField()
    architecture_overview: str = InputField()
    important_directories: List[str] = InputField()
    entry_points: List[str] = InputField()
    development_info: str = InputField()
    usage_examples: str = InputField()
    
    llms_txt_content: str = OutputField(desc="Complete llms.txt file content following standard format")
```

## Step 2: Create the Repository Analyzer Module

```python
# analyzer.py
from logillm.core.predict import Predict
from logillm.core.modules import Module
from signatures import AnalyzeRepository, AnalyzeCodeStructure, GenerateUsageExamples, GenerateLLMsTxt

class RepositoryAnalyzer(Module):
    """A comprehensive repository documentation generator using LogiLLM."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize our analysis components
        self.analyze_repo = Predict(signature=AnalyzeRepository)
        self.analyze_structure = Predict(signature=AnalyzeCodeStructure)
        self.generate_examples = Predict(signature=GenerateUsageExamples)
        self.generate_llms_txt = Predict(signature=GenerateLLMsTxt)
    
    async def forward(self, repo_url: str, file_tree: str, readme_content: str, package_files: str):
        """Process repository information and generate llms.txt documentation."""
        
        # Step 1: Analyze repository purpose and concepts
        repo_analysis = await self.analyze_repo(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=readme_content
        )
        
        # Step 2: Analyze code structure
        structure_analysis = await self.analyze_structure(
            file_tree=file_tree,
            package_files=package_files
        )
        
        # Step 3: Generate practical usage examples
        usage_examples = await self.generate_examples(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview
        )
        
        # Step 4: Generate final llms.txt
        llms_txt = await self.generate_llms_txt(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
            important_directories=structure_analysis.important_directories,
            entry_points=structure_analysis.entry_points,
            development_info=structure_analysis.development_info,
            usage_examples=usage_examples.usage_examples
        )
        
        return {
            'llms_txt_content': llms_txt.llms_txt_content,
            'analysis': repo_analysis,
            'structure': structure_analysis,
            'examples': usage_examples
        }
```

## Step 3: GitHub Repository Information Gathering

```python
# github_utils.py
import requests
import os
import base64
from typing import Optional

def get_github_file_tree(repo_url: str, github_token: Optional[str] = None) -> str:
    """Get repository file structure from GitHub API."""
    
    # Extract owner/repo from URL
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    
    headers = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        tree_data = response.json()
        file_paths = [
            item['path'] for item in tree_data['tree'] 
            if item['type'] == 'blob' and not item['path'].startswith('.')
        ]
        return '\n'.join(sorted(file_paths))
    else:
        raise Exception(f"Failed to fetch repository tree: {response.status_code}")

def get_github_file_content(repo_url: str, file_path: str, github_token: Optional[str] = None) -> str:
    """Get specific file content from GitHub."""
    
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    
    headers = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        content_b64 = response.json()['content']
        content = base64.b64decode(content_b64).decode('utf-8', errors='ignore')
        return content
    else:
        return f"Could not fetch {file_path}"

def gather_repository_info(repo_url: str, github_token: Optional[str] = None) -> tuple:
    """Gather all necessary repository information."""
    
    # Get file tree
    file_tree = get_github_file_tree(repo_url, github_token)
    
    # Get README content
    readme_content = get_github_file_content(repo_url, "README.md", github_token)
    
    # Get key package files
    package_files = []
    for file_path in ["pyproject.toml", "setup.py", "requirements.txt", "package.json", "Cargo.toml"]:
        try:
            content = get_github_file_content(repo_url, file_path, github_token)
            if "Could not fetch" not in content:
                package_files.append(f"=== {file_path} ===\n{content}")
        except:
            continue
    
    package_files_content = "\n\n".join(package_files) if package_files else "No package files found"
    
    return file_tree, readme_content, package_files_content
```

## Step 4: Main Application with Provider Setup

```python
# llms_txt_generator.py
import asyncio
import os
from pathlib import Path
from typing import Optional

from logillm.providers import create_provider, register_provider
from analyzer import RepositoryAnalyzer
from github_utils import gather_repository_info

async def generate_llms_txt_for_repo(
    repo_url: str,
    output_path: str = "llms.txt",
    github_token: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> None:
    """Generate llms.txt documentation for a GitHub repository."""
    
    print(f"🔍 Analyzing repository: {repo_url}")
    
    # Step 1: Setup LogiLLM provider
    if model.startswith("gpt"):
        provider = create_provider("openai", model=model)
    elif model.startswith("claude"):
        provider = create_provider("anthropic", model=model)
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    register_provider(provider, set_default=True)
    
    # Step 2: Initialize analyzer
    analyzer = RepositoryAnalyzer()
    
    # Step 3: Gather repository information
    print("📁 Gathering repository information...")
    file_tree, readme_content, package_files = gather_repository_info(repo_url, github_token)
    
    # Step 4: Generate documentation
    print("🤖 Generating llms.txt documentation...")
    result = await analyzer.forward(
        repo_url=repo_url,
        file_tree=file_tree,
        readme_content=readme_content,
        package_files=package_files
    )
    
    # Step 5: Save the generated documentation
    output_file = Path(output_path)
    output_file.write_text(result['llms_txt_content'])
    
    print(f"✅ Generated llms.txt saved to: {output_file.absolute()}")
    print(f"📊 Project: {result['analysis'].project_purpose[:100]}...")
    print(f"🏗️  Architecture: {result['analysis'].architecture_overview[:100]}...")
    
    # Step 6: Show preview
    print("\n📄 Preview (first 500 characters):")
    print("-" * 50)
    print(result['llms_txt_content'][:500] + "..." if len(result['llms_txt_content']) > 500 else result['llms_txt_content'])

async def main():
    """Main application entry point."""
    
    # Configuration
    repo_url = "https://github.com/stanfordnlp/dspy"  # Example repository
    github_token = os.environ.get("GITHUB_TOKEN")    # Optional
    model = "gpt-4o-mini"                            # or "claude-3-5-sonnet-20241022"
    
    await generate_llms_txt_for_repo(
        repo_url=repo_url,
        github_token=github_token,
        model=model
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 5: Advanced Features - Optimization and Persistence

```python
# optimization_example.py
import asyncio
from logillm.core.optimizers import AccuracyMetric
from logillm.optimizers import BootstrapFewShot
from analyzer import RepositoryAnalyzer

async def optimize_documentation_generator():
    """Demonstrate how to optimize the documentation generator for better results."""
    
    # Sample training data (in production, you'd have more examples)
    training_data = [
        {
            "inputs": {
                "repo_url": "https://github.com/example/simple-lib",
                "file_tree": "src/main.py\nREADME.md\nsetup.py",
                "readme_content": "# Simple Library\nA basic Python utility library.",
                "package_files": "setup.py content here"
            },
            "outputs": {
                "llms_txt_content": "# Simple Library\n\n## Overview\nA basic Python utility library...\n\n## Usage\n```python\nimport simple_lib\n```"
            }
        }
        # Add more training examples...
    ]
    
    # Create analyzer and optimize
    analyzer = RepositoryAnalyzer()
    
    # Define quality metric
    def documentation_quality_metric(prediction, reference):
        """Custom metric to evaluate documentation quality."""
        # In practice, you might use BLEU score, semantic similarity, or human evaluation
        pred_content = prediction.get('llms_txt_content', '')
        ref_content = reference.get('llms_txt_content', '')
        
        # Simple word overlap as a proxy for quality
        pred_words = set(pred_content.lower().split())
        ref_words = set(ref_content.lower().split())
        
        if not ref_words:
            return 0.0
        
        overlap = len(pred_words.intersection(ref_words))
        return overlap / len(ref_words)
    
    # Optimize the analyzer
    metric = AccuracyMetric(key="llms_txt_content", metric_fn=documentation_quality_metric)
    optimizer = BootstrapFewShot(metric=metric, max_examples=3)
    
    print("🎯 Optimizing documentation generator...")
    result = await optimizer.optimize(
        module=analyzer,
        dataset=training_data
    )
    
    # Save the optimized model
    optimized_analyzer = result.optimized_module
    optimized_analyzer.save("models/optimized_doc_generator.json")
    
    print(f"✅ Optimization complete! Improvement: {result.improvement:.2%}")
    print("💾 Optimized model saved to: models/optimized_doc_generator.json")
    
    return optimized_analyzer

async def use_optimized_generator():
    """Load and use the optimized documentation generator."""
    
    # Load the pre-trained, optimized model
    analyzer = RepositoryAnalyzer.load("models/optimized_doc_generator.json")
    
    # Use it for fast, high-quality documentation generation
    result = await analyzer.forward(
        repo_url="https://github.com/new-project/repo",
        file_tree="...",
        readme_content="...",
        package_files="..."
    )
    
    return result
```

## Testing the Tutorial

Create a test script to verify everything works:

```bash
# test_llms_txt_tutorial.py
"""
Test script for the llms.txt generation tutorial.
Run with: uv run --with logillm[openai] --with requests python test_llms_txt_tutorial.py
"""

import asyncio
import os
from llms_txt_generator import generate_llms_txt_for_repo

async def test_tutorial():
    """Test the llms.txt generation tutorial."""
    
    # Ensure API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return
    
    # Test with a small public repository
    test_repo = "https://github.com/octocat/Hello-World"
    
    try:
        await generate_llms_txt_for_repo(
            repo_url=test_repo,
            output_path="test_llms.txt",
            model="gpt-4o-mini"
        )
        print("✅ Tutorial test completed successfully!")
    
    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_tutorial())
```

## Expected Output Structure

The generated `llms.txt` for a repository will follow this structure:

```
# Project Name: Repository Documentation Framework

## Project Overview
LogiLLM is a high-performance, zero-dependency LLM programming framework inspired by DSPy...

## Key Concepts
- **Modules**: Building blocks for LLM applications
- **Signatures**: Input/output specifications with type hints
- **Providers**: LLM service integrations (OpenAI, Anthropic, etc.)
- **Optimization**: Hybrid prompt and hyperparameter tuning

## Architecture
- `/logillm/`: Main package directory
  - `/core/`: Core framework components
  - `/providers/`: LLM provider implementations
  - `/optimizers/`: Optimization algorithms
  - `/examples/`: Usage examples and demos

## Development Information
- Python 3.9+ required
- Zero core dependencies
- Optional provider dependencies
- Async-first design

## Usage Examples
1. **Basic Prediction**: Create a simple question-answering module
2. **Structured Extraction**: Extract structured data from unstructured text
3. **Optimization**: Improve model performance with training data
4. **Persistence**: Save and load optimized models for production
```

## Key LogiLLM Advantages Demonstrated

This tutorial showcases several LogiLLM advantages over DSPy:

1. **Clean Architecture**: No metaclass magic, explicit initialization
2. **Zero Dependencies**: Core framework requires no external packages
3. **Modern Python**: Full async/await support throughout
4. **Type Safety**: Complete type hints and IDE support
5. **Persistence**: Save/load optimized models for production use
6. **Hybrid Optimization**: Can optimize both prompts and hyperparameters

## 🎓 What You've Learned

Congratulations! You've mastered the fundamentals of LogiLLM:

✅ **Signatures**: Defined structured input/output with type safety  
✅ **Predict Modules**: Created LLM-powered processing components  
✅ **Provider Setup**: Configured OpenAI/Anthropic for production use  
✅ **External APIs**: Integrated GitHub for real-world data processing  
✅ **Module Composition**: Combined multiple signatures into a complete system

## 🚀 What's Next?

### Immediate Next Steps
**Ready for structured data?** → **[Email Extraction Tutorial](./email-extraction.md)**  
Learn how to transform the free-form text generation you just mastered into validated, structured data models using Pydantic.

### Apply What You've Learned
- **Try different repositories**: Process your own GitHub projects
- **Experiment with prompts**: Modify the signatures to generate different documentation styles
- **Add error handling**: Enhance the system with retry logic and graceful failures

### Advanced Extensions
- **Multiple repositories**: Process organization-wide documentation  
- **Quality metrics**: Add evaluation and scoring for generated docs
- **Web interface**: Build interactive repository analysis tools
- **CI/CD integration**: Automate documentation updates on code changes

### Tutorial Learning Path
1. ✅ **LLM Text Generation** (You are here!)
2. → **[Email Extraction](./email-extraction.md)** - Structured data processing
3. → **[Code Generation](./code-generation.md)** - Multi-step refinement  
4. → **[Yahoo Finance ReAct](./yahoo-finance-react.md)** - Agent reasoning
5. → **[AI Text Game](./ai-text-game.md)** - Interactive systems
6. → **[Memory-Enhanced ReAct Agent](./memory-enhanced-react.md)** - Persistent memory

## 🛠️ Running the Tutorial

```bash
# With OpenAI
export OPENAI_API_KEY="your-key-here"
uv run --with logillm --with openai --with requests python examples/tutorials/llms_txt_generation/llms_txt_generator.py

# With Anthropic
export ANTHROPIC_API_KEY="your-key-here" 
uv run --with logillm --with anthropic --with requests python examples/tutorials/llms_txt_generation/llms_txt_generator.py

# Run tests to verify your setup
uv run --with logillm --with openai --with requests python examples/tutorials/llms_txt_generation/test_tutorial.py
```

---

**📚 [← Back to Tutorial Index](./README.md) | [Next: Email Extraction →](./email-extraction.md)**

This tutorial established your LogiLLM foundation. Ready to learn structured data processing? Continue with **[Email Extraction](./email-extraction.md)** to build on these concepts!