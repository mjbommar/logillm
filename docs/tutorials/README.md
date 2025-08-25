# LogiLLM Tutorials

Welcome to LogiLLM! This comprehensive tutorial collection will take you from basic LLM text generation to building sophisticated AI agents with persistent memory.

## 🎯 Quick Start

### 🚀 Choose Your Path
- **[Getting Started Guide](./getting-started.md)** - Personalized tutorial recommendations
- **[Tutorial Matrix](./tutorial-matrix.md)** - Detailed difficulty and time analysis
- **Jump right in** with the path below

**New to LogiLLM?** Start here:
1. **[LLM Text Generation](./llms-txt-generation.md)** (10 min) - Learn the basics
2. **[Email Extraction](./email-extraction.md)** (15 min) - Structured data processing
3. **[Yahoo Finance ReAct](./yahoo-finance-react.md)** (20 min) - Your first agent

**Already familiar with DSPy?** Jump to **[Key Differences](#-key-differences-from-dspy)** below.

## 📚 Learning Paths

### 🌱 Beginner Path: Core Concepts
Perfect if you're new to LLM programming or coming from other frameworks.

1. **[LLM Text Generation](./llms-txt-generation.md)** 
   - ⏱️ 10-15 minutes | 🎯 Difficulty: Beginner
   - **Learn**: Signatures, Predict modules, provider setup
   - **Build**: Documentation generator that analyzes repositories
   - **Why Start Here**: Establishes LogiLLM fundamentals

2. **[Email Extraction](./email-extraction.md)**
   - ⏱️ 15-20 minutes | 🎯 Difficulty: Beginner  
   - **Learn**: Structured data extraction, Pydantic models, field validation
   - **Build**: Smart email processor with entity extraction
   - **Builds On**: Text generation → structured output

### 🚀 Intermediate Path: Real-World Applications  
Ready to build practical tools that solve actual problems.

3. **[Code Generation](./code-generation.md)**
   - ⏱️ 20-25 minutes | 🎯 Difficulty: Intermediate
   - **Learn**: Web scraping integration, iterative refinement, external APIs
   - **Build**: Learning assistant for unfamiliar libraries
   - **Builds On**: Structured extraction → multi-step processing

4. **[Yahoo Finance ReAct](./yahoo-finance-react.md)**  
   - ⏱️ 25-30 minutes | 🎯 Difficulty: Intermediate
   - **Learn**: ReAct pattern, tool integration, financial data APIs
   - **Build**: AI financial analyst with reasoning capabilities  
   - **Builds On**: Multi-step processing → reasoning agents

### 🎮 Advanced Path: Interactive AI Systems
Build sophisticated agents with memory and complex interactions.

5. **[AI Text Game](./ai-text-game.md)**
   - ⏱️ 20-25 minutes | 🎯 Difficulty: Intermediate-Advanced
   - **Learn**: Interactive systems, game state management, dynamic narratives
   - **Build**: AI dungeon master for text adventures
   - **Builds On**: Reasoning → interactive, stateful systems

6. **[Memory-Enhanced ReAct Agent](./memory-enhanced-react.md)**
   - ⏱️ 30-40 minutes | 🎯 Difficulty: Advanced  
   - **Learn**: Persistent memory, user profiles, conversation context
   - **Build**: Personal assistant that remembers across sessions
   - **Builds On**: Interactive systems → persistent, personalized AI

## 🎨 Learning by Use Case

### 📝 Text Processing & Generation
- **[LLM Text Generation](./llms-txt-generation.md)** - Repository documentation
- **[Code Generation](./code-generation.md)** - Learning new libraries

### 🔍 Data Extraction & Analysis  
- **[Email Extraction](./email-extraction.md)** - Structured email processing
- **[Yahoo Finance ReAct](./yahoo-finance-react.md)** - Financial data analysis

### 🤖 AI Agents & Interaction
- **[Yahoo Finance ReAct](./yahoo-finance-react.md)** - Reasoning agents
- **[AI Text Game](./ai-text-game.md)** - Interactive storytelling  
- **[Memory-Enhanced ReAct Agent](./memory-enhanced-react.md)** - Persistent assistants

## 💡 Tutorial Feature Matrix

| Tutorial | Core Concepts | Advanced Features | Real-World Skills |
|----------|---------------|------------------|-------------------|
| **LLM Text Generation** | Signatures, Predict, Providers | GitHub API, Repository Analysis | Documentation automation |
| **Email Extraction** | Pydantic Models, Field Validation | Classification, Priority Detection | Email processing systems |
| **Code Generation** | Web Scraping, Iterative Refinement | Documentation Analysis | Learning automation |
| **Yahoo Finance ReAct** | ReAct Pattern, Tool Integration | Financial APIs, Market Analysis | Financial AI assistants |
| **AI Text Game** | State Management, Interactive Systems | Dynamic Narratives, Game Logic | Interactive applications |
| **Memory-Enhanced ReAct** | Persistent Memory, User Profiles | Conversation Context, Memory Search | Personal AI assistants |

## 🔧 Technical Concepts Progression

### Level 1: Foundation
- **Signatures**: Define input/output structure with type safety
- **Predict Modules**: Basic LLM interaction patterns  
- **Provider Setup**: OpenAI, Anthropic, model selection

### Level 2: Structured Processing
- **Pydantic Integration**: Type-safe data models
- **Field Validation**: Input/output constraints and descriptions
- **Error Handling**: Graceful failure and retry patterns

### Level 3: External Integration  
- **Web Scraping**: Fetch and process external data
- **API Integration**: GitHub, Yahoo Finance, custom APIs
- **Multi-step Processing**: Chain operations for complex tasks

### Level 4: Agent Patterns
- **ReAct Framework**: Reasoning + Acting for complex decisions
- **Tool Integration**: Give agents access to external functions
- **Planning & Execution**: Break down complex requests

### Level 5: Persistent Systems
- **Memory Management**: Store and retrieve conversation history
- **User Profiles**: Personalized, multi-session interactions  
- **State Persistence**: Maintain context across application restarts

## 🎯 Key Differences from DSPy

If you're coming from DSPy, here are the key advantages you'll discover:

### 🚀 **Zero Dependencies**
- **DSPy**: Requires external services (Mem0, Redis) for memory
- **LogiLLM**: Built-in persistence, no external setup needed

### ⚡ **Modern Python Patterns**  
- **DSPy**: Callback-based, older Python patterns
- **LogiLLM**: Full async/await, modern type hints

### 🔒 **Type Safety Throughout**
- **DSPy**: Limited type checking, runtime errors common
- **LogiLLM**: Full Pydantic integration, catch errors at development time

### 📦 **Simplified Architecture**
- **DSPy**: Complex module hierarchy, unclear patterns  
- **LogiLLM**: Clean Signature + Predict + Module pattern

### 🎮 **Better Developer Experience**
- **DSPy**: Verbose setup, unclear error messages
- **LogiLLM**: Minimal boilerplate, clear error handling

## 🛠️ Quick Setup for All Tutorials

```bash
# Install dependencies
uv add logillm pydantic

# Set up API keys (choose one)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here" 

# Test your setup
uv run --with logillm --with openai python -c "
from logillm.providers import create_provider, register_provider
provider = create_provider('openai')
register_provider(provider, set_default=True)
print('✅ LogiLLM setup successful!')
"
```

## 🎓 Learning Tips

### Before You Start
- **Choose your LLM provider**: OpenAI (gpt-4o-mini) or Anthropic (claude-3-haiku) work great
- **Set up your environment**: Use `uv` for dependency management (much faster than pip)
- **Time expectations**: Each tutorial includes realistic time estimates

### As You Learn  
- **Run the tests**: Every tutorial has a test script (`test_tutorial.py` or `test_demo.py`) - use it to verify your setup
- **Experiment**: Try modifying examples with your own data/use cases
- **Read the signatures**: Understanding input/output definitions is key to LogiLLM

### After Each Tutorial
- **Check cross-references**: See how concepts connect to other tutorials
- **Try the extensions**: Each tutorial suggests next steps and improvements
- **Build something new**: Apply the patterns to your own problems

## 🔄 Tutorial Cross-References

### From Basic to Advanced Concepts

**LLM Text Generation** → **Email Extraction**: _Structured output_
- Learn how to go from free-form text to validated data models

**Email Extraction** → **Code Generation**: _Multi-step processing_  
- Build on structured data with iterative refinement and external APIs

**Code Generation** → **Yahoo Finance ReAct**: _Agent reasoning_
- Evolve from scripted processing to intelligent decision-making

**Yahoo Finance ReAct** → **AI Text Game**: _Interactive systems_
- Apply agent patterns to dynamic, user-driven interactions

**AI Text Game** → **Memory-Enhanced ReAct**: _Persistent context_
- Add memory and personalization to your interactive agents

### By Technical Skill

**Want to learn async patterns?** Start with Email Extraction, advance to Memory-Enhanced ReAct

**Want to understand agents?** Begin with Yahoo Finance ReAct, then AI Text Game

**Want external API integration?** Try Code Generation, then Yahoo Finance ReAct  

**Want persistent systems?** Go straight to Memory-Enhanced ReAct after basics

## 🤝 Getting Help

- **Stuck on setup?** Check the test file (`test_tutorial.py` or `test_demo.py`) in each tutorial
- **API errors?** Verify your API keys and try the simpler tutorials first  
- **Want to contribute?** All tutorials are in `/examples/tutorials/` - improvements welcome!
- **Found a bug?** Each tutorial is fully tested - let us know if something breaks

## 🎉 What's Next After Tutorials?

Once you've completed the tutorials, you're ready to:

1. **Build Production Apps**: Deploy your LogiLLM agents with FastAPI/Flask
2. **Custom Providers**: Add support for new LLM services  
3. **Advanced Patterns**: Multi-agent systems, streaming, real-time processing
4. **Contribute**: Help expand the tutorial collection!

---

**Ready to start?** Jump into **[LLM Text Generation](./llms-txt-generation.md)** and begin your LogiLLM journey! 🚀