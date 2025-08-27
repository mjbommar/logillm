"""LogiLLM signatures for financial analysis."""

from typing import Any

from logillm.core.signatures import InputField, OutputField, Signature


class AnalyzeStock(Signature):
    """Analyze a single stock's performance and provide insights."""

    stock_data: dict[str, Any] = InputField(desc="Stock price and company data")
    news_data: dict[str, Any] = InputField(desc="Recent news about the stock")

    analysis: str = OutputField(desc="Comprehensive analysis of the stock's current situation")
    recommendation: str = OutputField(
        desc="Investment recommendation (Buy/Hold/Sell) with reasoning"
    )
    risk_factors: str = OutputField(
        desc='Key risk factors as JSON array: ["risk1", "risk2", "risk3"]'
    )
    price_targets: str = OutputField(
        desc='Price targets as JSON object: {"1_month": 150.0, "3_month": 160.0, "1_year": 180.0}'
    )


class CompareStocks(Signature):
    """Compare multiple stocks and provide relative analysis."""

    comparison_data: dict[str, Any] = InputField(desc="Comparison data for multiple stocks")
    analysis_context: str = InputField(desc="Context for the comparison")

    comparison_summary: str = OutputField(desc="Summary of how the stocks compare")
    rankings: str = OutputField(
        desc='Stocks ranked as JSON array: ["AAPL: Strong buy", "MSFT: Hold", "etc"]'
    )
    investment_strategy: str = OutputField(desc="Suggested investment strategy based on comparison")


class FinancialQuery(Signature):
    """Process natural language financial queries and plan analysis steps."""

    user_query: str = InputField(desc="User's financial question or request")
    available_tools: list[str] = InputField(desc="List of available financial analysis tools")

    analysis_plan: str = OutputField(
        desc='Step-by-step plan as JSON array: ["Step 1: Get stock price", "Step 2: Analyze trends", "etc"]'
    )
    required_data: str = OutputField(
        desc='Required data as JSON array: ["current_price", "historical_data", "news"]'
    )
    expected_output: str = OutputField(desc="Description of expected final answer format")
