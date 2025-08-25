# Financial Analysis with LogiLLM ReAct and Yahoo Finance

> **📍 Tutorial Path**: [Code Generation](./code-generation.md) → **Yahoo Finance ReAct** → [AI Text Game](./ai-text-game.md) → [Memory-Enhanced ReAct](./memory-enhanced-react.md)  
> **⏱️ Time**: 25-30 minutes | **🎯 Difficulty**: Intermediate  
> **💡 Concepts**: ReAct pattern, Tool integration, Financial data APIs, Agent reasoning

This tutorial demonstrates how to build a financial analysis agent using LogiLLM's ReAct (Reasoning + Acting) framework. We'll create an intelligent agent that can retrieve real-time market data, analyze financial information, and provide insights through step-by-step reasoning.

**Perfect for**: Developers ready to learn agent patterns, anyone building financial tools, those wanting to understand ReAct architecture.

**Builds on**: [Code Generation](./code-generation.md) - Now we'll take multi-step processing and transform it into intelligent agents that can reason and act.

## What You'll Build

By the end of this tutorial, you'll have a LogiLLM-powered financial agent that can:

- **Retrieve stock prices** and financial data in real-time
- **Analyze market trends** and provide insights
- **Compare multiple stocks** side-by-side
- **Reason through complex queries** using the ReAct pattern
- **Handle financial calculations** and present results clearly
- **Provide investment recommendations** based on data analysis

## Prerequisites

- Python 3.9+ installed
- OpenAI or Anthropic API key
- Basic understanding of LogiLLM modules and signatures
- Familiarity with financial concepts (helpful but not required)

## Installation and Setup

```bash
# Install LogiLLM with provider support
pip install logillm[openai]

# For financial data retrieval
pip install yfinance requests

# For data analysis (optional)
pip install pandas numpy
```

## Step 1: Define Financial Tool Functions

First, let's create the core financial data retrieval functions:

```python
# financial_tools.py
import yfinance as yf
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json


def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get current stock price and basic information."""
    try:
        stock = yf.Ticker(symbol.upper())
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        previous_close = info.get('previousClose')
        
        if current_price is None:
            # Fallback to history data
            history = stock.history(period='1d')
            if not history.empty:
                current_price = float(history['Close'].iloc[-1])
                previous_close = float(history['Close'].iloc[-2]) if len(history) > 1 else current_price
        
        change = current_price - previous_close if (current_price and previous_close) else 0
        change_percent = (change / previous_close * 100) if previous_close else 0
        
        return {
            'symbol': symbol.upper(),
            'current_price': current_price,
            'previous_close': previous_close,
            'change': change,
            'change_percent': change_percent,
            'company_name': info.get('longName', symbol.upper()),
            'market_cap': info.get('marketCap'),
            'volume': info.get('volume'),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'symbol': symbol.upper(),
            'status': 'error',
            'error': str(e)
        }


def compare_stocks(symbols: List[str]) -> Dict[str, Any]:
    """Compare multiple stocks side by side."""
    results = {}
    
    for symbol in symbols:
        results[symbol] = get_stock_price(symbol)
    
    # Calculate relative performance
    successful_stocks = {k: v for k, v in results.items() if v['status'] == 'success'}
    
    if len(successful_stocks) > 1:
        best_performer = max(successful_stocks.keys(), 
                           key=lambda x: successful_stocks[x].get('change_percent', -float('inf')))
        worst_performer = min(successful_stocks.keys(),
                            key=lambda x: successful_stocks[x].get('change_percent', float('inf')))
        
        return {
            'comparison': results,
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'comparison_summary': f"{best_performer} is the best performer, {worst_performer} is the worst"
        }
    
    return {'comparison': results}


def get_stock_news(symbol: str, max_articles: int = 5) -> Dict[str, Any]:
    """Get recent news for a stock (mock implementation)."""
    # In a real implementation, you'd use a news API like Alpha Vantage or NewsAPI
    # This is a mock implementation for demonstration
    
    mock_news = [
        {
            'title': f'{symbol} Reports Strong Q4 Earnings',
            'summary': f'{symbol} exceeded analyst expectations with strong revenue growth.',
            'source': 'Financial Times',
            'timestamp': datetime.now().isoformat()
        },
        {
            'title': f'Analysts Upgrade {symbol} Rating',
            'summary': f'Major investment firm upgrades {symbol} to "Buy" rating.',
            'source': 'Bloomberg',
            'timestamp': (datetime.now() - timedelta(hours=2)).isoformat()
        },
        {
            'title': f'{symbol} Announces New Product Launch',
            'summary': f'{symbol} unveils innovative product expected to drive growth.',
            'source': 'Reuters',
            'timestamp': (datetime.now() - timedelta(hours=4)).isoformat()
        }
    ]
    
    return {
        'symbol': symbol.upper(),
        'news': mock_news[:max_articles],
        'status': 'success'
    }


def calculate_financial_metrics(price_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate basic financial metrics from price data."""
    if price_data.get('status') != 'success':
        return {'error': 'Invalid price data'}
    
    current_price = price_data.get('current_price', 0)
    market_cap = price_data.get('market_cap', 0)
    volume = price_data.get('volume', 0)
    
    return {
        'symbol': price_data['symbol'],
        'current_price': current_price,
        'market_cap_billions': market_cap / 1e9 if market_cap else None,
        'volume_millions': volume / 1e6 if volume else None,
        'price_tier': 'High' if current_price > 100 else 'Medium' if current_price > 20 else 'Low'
    }
```

## Step 2: Create LogiLLM Signatures for Financial Analysis

```python
# signatures.py
from logillm.core.signatures import Signature, InputField, OutputField
from typing import List, Dict, Any, Optional


class AnalyzeStock(Signature):
    """Analyze a single stock's performance and provide insights."""
    
    stock_data: Dict[str, Any] = InputField(desc="Stock price and company data")
    news_data: Dict[str, Any] = InputField(desc="Recent news about the stock")
    
    analysis: str = OutputField(desc="Comprehensive analysis of the stock's current situation")
    recommendation: str = OutputField(desc="Investment recommendation (Buy/Hold/Sell) with reasoning")
    risk_factors: List[str] = OutputField(desc="Key risk factors to consider")
    price_targets: Dict[str, float] = OutputField(desc="Price targets for different time horizons")


class CompareStocks(Signature):
    """Compare multiple stocks and provide relative analysis."""
    
    comparison_data: Dict[str, Any] = InputField(desc="Comparison data for multiple stocks")
    analysis_context: str = InputField(desc="Context for the comparison (e.g., 'tech stocks', 'dividend stocks')")
    
    comparison_summary: str = OutputField(desc="Summary of how the stocks compare")
    rankings: List[str] = OutputField(desc="Stocks ranked from best to worst with reasoning")
    investment_strategy: str = OutputField(desc="Suggested investment strategy based on comparison")


class FinancialQuery(Signature):
    """Process natural language financial queries and plan analysis steps."""
    
    user_query: str = InputField(desc="User's financial question or request")
    available_tools: List[str] = InputField(desc="List of available financial analysis tools")
    
    analysis_plan: List[str] = OutputField(desc="Step-by-step plan to answer the query")
    required_data: List[str] = OutputField(desc="Data that needs to be retrieved")
    expected_output: str = OutputField(desc="Description of expected final answer format")
```

## Step 3: Build the Financial ReAct Agent

```python
# agent.py
from typing import Dict, List, Any, Optional, Callable
from logillm.core.predict import Predict
from logillm.core.modules import Module
from .signatures import AnalyzeStock, CompareStocks, FinancialQuery
from .financial_tools import get_stock_price, compare_stocks, get_stock_news, calculate_financial_metrics


class FinancialReActAgent(Module):
    """ReAct agent for financial analysis using LogiLLM."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Initialize LogiLLM components
        self.query_planner = Predict(signature=FinancialQuery)
        self.stock_analyzer = Predict(signature=AnalyzeStock)
        self.stock_comparator = Predict(signature=CompareStocks)
        
        # Available tools
        self.tools = {
            'get_stock_price': get_stock_price,
            'compare_stocks': compare_stocks,
            'get_stock_news': get_stock_news,
            'calculate_financial_metrics': calculate_financial_metrics
        }
    
    async def plan_analysis(self, user_query: str) -> Any:
        """Plan how to approach a financial query."""
        available_tools = list(self.tools.keys())
        
        return await self.query_planner(
            user_query=user_query,
            available_tools=available_tools
        )
    
    async def analyze_single_stock(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a single stock."""
        
        # Step 1: Get stock data
        print(f"📈 Getting stock data for {symbol}...")
        stock_data = get_stock_price(symbol)
        
        if stock_data['status'] != 'success':
            return {'error': f"Could not retrieve data for {symbol}"}
        
        # Step 2: Get news
        print(f"📰 Getting news for {symbol}...")
        news_data = get_stock_news(symbol)
        
        # Step 3: Analyze with LogiLLM
        print(f"🤖 Analyzing {symbol}...")
        analysis = await self.stock_analyzer(
            stock_data=stock_data,
            news_data=news_data
        )
        
        return {
            'symbol': symbol,
            'stock_data': stock_data,
            'news_data': news_data,
            'analysis': analysis
        }
    
    async def compare_multiple_stocks(self, symbols: List[str], context: str = "") -> Dict[str, Any]:
        """Compare multiple stocks and provide ranking."""
        
        print(f"📊 Comparing stocks: {', '.join(symbols)}...")
        comparison_data = compare_stocks(symbols)
        
        print(f"🤖 Analyzing comparison...")
        comparison = await self.stock_comparator(
            comparison_data=comparison_data,
            analysis_context=context
        )
        
        return {
            'symbols': symbols,
            'comparison_data': comparison_data,
            'comparison_analysis': comparison
        }
    
    async def forward(self, user_query: str) -> Dict[str, Any]:
        """Process a financial query using ReAct methodology."""
        
        print(f"🎯 Processing query: {user_query}")
        
        # Step 1: Plan the analysis
        plan = await self.plan_analysis(user_query)
        
        print(f"📋 Analysis plan:")
        for i, step in enumerate(plan.analysis_plan, 1):
            print(f"   {i}. {step}")
        
        # Step 2: Execute the plan (simplified ReAct loop)
        results = {}
        
        # Extract symbols from query (simple approach)
        words = user_query.upper().split()
        potential_symbols = [word for word in words if len(word) <= 5 and word.isalpha()]
        
        # Common stock symbols for fallback
        if not potential_symbols:
            if any(term in user_query.lower() for term in ['apple', 'aapl']):
                potential_symbols = ['AAPL']
            elif any(term in user_query.lower() for term in ['microsoft', 'msft']):
                potential_symbols = ['MSFT']
            elif any(term in user_query.lower() for term in ['google', 'googl']):
                potential_symbols = ['GOOGL']
            elif any(term in user_query.lower() for term in ['compare', 'vs']):
                potential_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Default comparison
        
        if not potential_symbols:
            return {
                'error': 'Could not identify stock symbols in query. Please include specific stock symbols.',
                'suggestion': 'Try queries like "Analyze AAPL" or "Compare AAPL vs MSFT"'
            }
        
        # Step 3: Execute based on query type
        if len(potential_symbols) == 1:
            results = await self.analyze_single_stock(potential_symbols[0])
        else:
            context = "general comparison"
            if any(term in user_query.lower() for term in ['tech', 'technology']):
                context = "technology stocks"
            elif any(term in user_query.lower() for term in ['dividend']):
                context = "dividend stocks"
            
            results = await self.compare_multiple_stocks(potential_symbols, context)
        
        # Step 4: Return comprehensive results
        return {
            'query': user_query,
            'plan': plan,
            'results': results,
            'symbols_analyzed': potential_symbols
        }
```

## Step 4: Demo Application

```python
# demo.py
import asyncio
import os
from typing import List
from logillm.providers import create_provider, register_provider
from .agent import FinancialReActAgent


async def demo_financial_analysis() -> None:
    """Demonstrate the financial analysis agent."""
    
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
    
    # Create financial agent
    agent = FinancialReActAgent()
    
    # Sample queries
    queries = [
        "Analyze AAPL stock performance and provide investment recommendation",
        "Compare AAPL vs MSFT for technology investment",
        "What is the current price of GOOGL and should I buy it?"
    ]
    
    print("🚀 Financial Analysis Agent Demo")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n💼 QUERY {i}: {query}")
        print("-" * 60)
        
        try:
            result = await agent.forward(query)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                if 'suggestion' in result:
                    print(f"💡 Suggestion: {result['suggestion']}")
                continue
            
            # Display results
            symbols = result.get('symbols_analyzed', [])
            print(f"📊 Analyzed symbols: {', '.join(symbols)}")
            
            if 'results' in result:
                results = result['results']
                
                # Single stock analysis
                if 'analysis' in results:
                    analysis = results['analysis']
                    stock_data = results['stock_data']
                    
                    print(f"\n💰 Current Price: ${stock_data.get('current_price', 'N/A')}")
                    print(f"📈 Change: {stock_data.get('change', 0):.2f} ({stock_data.get('change_percent', 0):.2f}%)")
                    print(f"🏢 Company: {stock_data.get('company_name', 'N/A')}")
                    
                    print(f"\n🔍 Analysis:")
                    print(f"   {analysis.analysis}")
                    
                    print(f"\n💡 Recommendation:")
                    print(f"   {analysis.recommendation}")
                    
                    if analysis.risk_factors:
                        print(f"\n⚠️  Risk Factors:")
                        for risk in analysis.risk_factors:
                            print(f"   • {risk}")
                
                # Stock comparison
                elif 'comparison_analysis' in results:
                    comparison = results['comparison_analysis']
                    comparison_data = results['comparison_data']['comparison']
                    
                    print(f"\n📊 Stock Comparison:")
                    for symbol, data in comparison_data.items():
                        if data['status'] == 'success':
                            print(f"   {symbol}: ${data.get('current_price', 'N/A')} ({data.get('change_percent', 0):.2f}%)")
                    
                    print(f"\n📝 Comparison Summary:")
                    print(f"   {comparison.comparison_summary}")
                    
                    print(f"\n🏆 Rankings:")
                    for i, ranking in enumerate(comparison.rankings, 1):
                        print(f"   {i}. {ranking}")
                    
                    print(f"\n💼 Investment Strategy:")
                    print(f"   {comparison.investment_strategy}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()


async def main() -> None:
    """Main demo entry point."""
    await demo_financial_analysis()


if __name__ == "__main__":
    asyncio.run(main())
```

## Testing the Tutorial

```python
# test_tutorial.py
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.tutorials.yahoo_finance_react.demo import demo_financial_analysis


async def test_tutorial() -> None:
    """Test the financial analysis tutorial."""
    
    model = os.environ.get("MODEL", "gpt-4o-mini")
    
    if model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return
    elif model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        return
    
    try:
        print("🧪 Running financial analysis tutorial test...")
        await demo_financial_analysis()
        print("✅ Tutorial test completed successfully!")
        
    except Exception as e:
        print(f"❌ Tutorial test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tutorial())
```

## Key Features Demonstrated

This tutorial showcases:

1. **ReAct Pattern**: Step-by-step reasoning combined with tool execution
2. **Real-time Data Integration**: Live financial data retrieval
3. **Multi-step Analysis**: Complex queries broken down into manageable steps
4. **Tool Integration**: External APIs seamlessly integrated with LogiLLM
5. **Structured Output**: Clean, organized analysis results
6. **Error Handling**: Robust handling of API failures and invalid inputs

## 🎓 What You've Learned

Fantastic! You've now mastered agent-based reasoning systems:

✅ **ReAct Pattern**: Built agents that reason through problems step-by-step  
✅ **Tool Integration**: Connected LLMs to external APIs and data sources  
✅ **Agent Planning**: Created systems that plan and execute multi-step workflows  
✅ **Real-time Data**: Integrated live financial data with AI analysis  
✅ **Decision Making**: Built agents that can analyze and recommend actions

## 🚀 What's Next?

### Immediate Next Steps
**Ready for interactive systems?** → **[AI Text Game Tutorial](./ai-text-game.md)**  
Learn how to take the agent reasoning you just mastered and apply it to dynamic, user-driven interactive experiences.

### Apply What You've Learned
- **Expand to more markets**: Add cryptocurrency, forex, commodities data
- **Add more analysis tools**: Technical indicators, sentiment analysis, news integration
- **Build alerts system**: Notifications based on price movements or analysis

### Advanced Extensions
- **Portfolio management**: Track and analyze investment portfolios
- **Risk assessment**: Add volatility and risk analysis tools
- **Trading strategies**: Implement backtesting and strategy evaluation
- **Real-time dashboards**: Build live monitoring interfaces

### Tutorial Learning Path
1. ✅ **[LLM Text Generation](./llms-txt-generation.md)** - Foundation concepts
2. ✅ **[Email Extraction](./email-extraction.md)** - Structured data processing
3. ✅ **[Code Generation](./code-generation.md)** - Multi-step processing  
4. ✅ **Yahoo Finance ReAct** (You are here!)
5. → **[AI Text Game](./ai-text-game.md)** - Interactive systems
6. → **[Memory-Enhanced ReAct Agent](./memory-enhanced-react.md)** - Persistent memory

### Concept Connections
- **From ReAct to Interactive Games**: Agent reasoning becomes the foundation for dynamic storytelling
- **Tool Patterns**: The financial tools you built are templates for any external API integration
- **Planning & Execution**: These patterns scale to complex multi-agent systems

## 🛠️ Running the Tutorial

```bash
# With OpenAI
export OPENAI_API_KEY="your-key-here"
uv run --with logillm --with openai --with yfinance python -m examples.tutorials.yahoo_finance_react.demo

# With Anthropic
export ANTHROPIC_API_KEY="your-key-here"
uv run --with logillm --with anthropic --with yfinance python -m examples.tutorials.yahoo_finance_react.demo

# Run tests to verify your setup
uv run --with logillm --with openai --with yfinance python examples/tutorials/yahoo_finance_react/test_tutorial.py
```

---

**📚 [← Code Generation](./code-generation.md) | [Tutorial Index](./README.md) | [AI Text Game →](./ai-text-game.md)**

You've mastered agent reasoning! Ready to build interactive systems? Continue with **[AI Text Game](./ai-text-game.md)** to apply these concepts to dynamic user interactions.