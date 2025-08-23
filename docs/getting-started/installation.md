# Installation Guide

This guide covers installing LogiLLM and its optional dependencies.

## Requirements

- Python 3.13 or higher
- pip or uv package manager

## Core Installation

LogiLLM's core functionality requires no external dependencies:

```bash
pip install logillm
```

This gives you access to all core features including:
- Signature system
- Module composition
- Optimization algorithms
- Mock provider for testing

## Installing with Providers

LLM providers are optional dependencies. Install only what you need:

### OpenAI

```bash
pip install logillm[openai]
```

Requires environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

### Anthropic (Claude)

```bash
pip install logillm[anthropic]
```

Requires environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### Google (Gemini)

```bash
pip install logillm[google]
```

Requires environment variable:
```bash
export GOOGLE_API_KEY="your-api-key"
```

### All Providers

```bash
pip install logillm[all]
```

## Installation with uv

If you're using uv for package management:

```bash
# Core installation
uv add logillm

# With providers
uv add logillm --optional openai
uv add logillm --optional anthropic
uv add logillm --optional all
```

## Development Installation

To contribute to LogiLLM or run tests:

```bash
# Clone the repository
git clone https://github.com/yourusername/logillm.git
cd logillm

# Install with development dependencies
pip install -e ".[dev]"

# Or with uv
uv pip install -e . --all-extras
```

## Verifying Installation

Check that LogiLLM is installed correctly:

```python
import logillm
print(logillm.__version__)

# Test basic functionality
from logillm import Predict
qa = Predict("question -> answer")
print("Installation successful!")
```

## Docker Installation

For containerized deployments:

```dockerfile
FROM python:3.13-slim

# Install LogiLLM with minimal footprint
RUN pip install logillm

# Add providers as needed
RUN pip install logillm[openai]

# Your application
COPY app.py .
CMD ["python", "app.py"]
```

## Troubleshooting

### Import Errors

If you get import errors for providers:

```python
# This will fail without the openai extra
from logillm.providers import OpenAIProvider  # ImportError

# Solution: Install the provider
pip install logillm[openai]
```

### Environment Variables

Providers look for API keys in environment variables:

```python
import os

# Set programmatically if needed
os.environ["OPENAI_API_KEY"] = "your-key"

# Or load from .env file
from dotenv import load_dotenv
load_dotenv()
```

### Python Version

LogiLLM requires Python 3.13+. Check your version:

```bash
python --version
# Should show: Python 3.13.x or higher
```

## Next Steps

- Continue to the [Quickstart Guide](quickstart.md) to build your first application
- Read about [Core Concepts](../core-concepts/philosophy.md) to understand LogiLLM's design
- Explore [Provider Configuration](../providers/openai.md) for detailed setup