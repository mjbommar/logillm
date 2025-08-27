"""Financial ReAct agent implementation."""

import json
from typing import Any

from logillm.core.modules import Module
from logillm.core.predict import Predict

from .financial_tools import compare_stocks, get_stock_news, get_stock_price
from .signatures import AnalyzeStock, CompareStocks, FinancialQuery


class FinancialReActAgent(Module):
    """ReAct agent for financial analysis using LogiLLM."""

    def __init__(self) -> None:
        super().__init__()

        self.query_planner = Predict(signature=FinancialQuery)
        self.stock_analyzer = Predict(signature=AnalyzeStock)
        self.stock_comparator = Predict(signature=CompareStocks)

        self.tools = {
            "get_stock_price": get_stock_price,
            "compare_stocks": compare_stocks,
            "get_stock_news": get_stock_news,
        }

    def _parse_json_list(self, text: str) -> list[str]:
        """Parse JSON array from text, with fallback to splitting."""
        if not text:
            return []

        text = text.strip()

        # Try JSON array format
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Fallback: split by comma or newlines
        if "," in text:
            return [item.strip() for item in text.split(",")]
        elif "\n" in text:
            return [item.strip() for item in text.split("\n") if item.strip()]

        return [text] if text else []

    def _parse_json_dict(self, text: str) -> dict[str, float]:
        """Parse JSON object from text."""
        if not text:
            return {}

        text = text.strip()

        # Try JSON object format
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Fallback: return empty dict
        return {}

    async def plan_analysis(self, user_query: str) -> Any:
        """Plan how to approach a financial query."""
        return await self.query_planner(
            user_query=user_query, available_tools=list(self.tools.keys())
        )

    async def analyze_single_stock(self, symbol: str) -> dict[str, Any]:
        """Perform comprehensive analysis of a single stock."""
        print(f"📈 Getting stock data for {symbol}...")
        stock_data = get_stock_price(symbol)

        if stock_data["status"] != "success":
            return {"error": f"Could not retrieve data for {symbol}"}

        print(f"📰 Getting news for {symbol}...")
        news_data = get_stock_news(symbol)

        print(f"🤖 Analyzing {symbol}...")
        analysis = await self.stock_analyzer(stock_data=stock_data, news_data=news_data)

        return {
            "symbol": symbol,
            "stock_data": stock_data,
            "news_data": news_data,
            "analysis": analysis,
        }

    async def compare_multiple_stocks(
        self, symbols: list[str], context: str = ""
    ) -> dict[str, Any]:
        """Compare multiple stocks and provide ranking."""
        print(f"📊 Comparing stocks: {', '.join(symbols)}...")
        comparison_data = compare_stocks(symbols)

        print("🤖 Analyzing comparison...")
        comparison = await self.stock_comparator(
            comparison_data=comparison_data, analysis_context=context
        )

        return {
            "symbols": symbols,
            "comparison_data": comparison_data,
            "comparison_analysis": comparison,
        }

    async def forward(self, user_query: str) -> dict[str, Any]:
        """Process a financial query using ReAct methodology."""
        print(f"🎯 Processing query: {user_query}")

        plan = await self.plan_analysis(user_query)

        print("📋 Analysis plan:")
        parsed_plan = self._parse_json_list(plan.analysis_plan)
        for i, step in enumerate(parsed_plan, 1):
            print(f"   {i}. {step}")

        # Extract symbols from query (simplified)
        words = user_query.upper().split()
        potential_symbols = [word for word in words if len(word) <= 5 and word.isalpha()]

        if not potential_symbols:
            if any(term in user_query.lower() for term in ["apple", "aapl"]):
                potential_symbols = ["AAPL"]
            elif any(term in user_query.lower() for term in ["microsoft", "msft"]):
                potential_symbols = ["MSFT"]
            elif any(term in user_query.lower() for term in ["compare", "vs"]):
                potential_symbols = ["AAPL", "MSFT"]

        if not potential_symbols:
            return {
                "error": "Could not identify stock symbols in query.",
                "suggestion": 'Try queries like "Analyze AAPL" or "Compare AAPL vs MSFT"',
            }

        if len(potential_symbols) == 1:
            results = await self.analyze_single_stock(potential_symbols[0])
        else:
            context = "general comparison"
            if any(term in user_query.lower() for term in ["tech"]):
                context = "technology stocks"
            results = await self.compare_multiple_stocks(potential_symbols, context)

        return {
            "query": user_query,
            "plan": plan,
            "results": results,
            "symbols_analyzed": potential_symbols,
        }
