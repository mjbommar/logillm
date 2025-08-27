"""Demo application for Yahoo Finance ReAct tutorial."""

import asyncio
import json
import os

from logillm.providers import create_provider, register_provider

from .agent import FinancialReActAgent


def _parse_json_list(text: str) -> list[str]:
    """Helper to parse JSON list with fallbacks."""
    if not text:
        return []

    text = text.strip()

    # Try JSON array format
    if text.startswith("[") and text.endswith("]"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Fallback: split by comma
    if "," in text:
        return [item.strip() for item in text.split(",")]

    return [text] if text else []


async def demo_financial_analysis() -> None:
    """Demonstrate the financial analysis agent."""
    model = os.environ.get("MODEL", "gpt-4.1")

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

    agent = FinancialReActAgent()

    queries = ["Analyze AAPL stock performance", "Compare AAPL vs MSFT"]

    print("🚀 Financial Analysis Agent Demo")
    print("=" * 50)

    for i, query in enumerate(queries, 1):
        print(f"\n💼 QUERY {i}: {query}")
        print("-" * 60)

        try:
            result = await agent.forward(query)

            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue

            symbols = result.get("symbols_analyzed", [])
            print(f"📊 Analyzed symbols: {', '.join(symbols)}")

            if "results" in result:
                results = result["results"]

                if "analysis" in results:
                    analysis = results["analysis"]
                    stock_data = results["stock_data"]

                    print(f"\n💰 Current Price: ${stock_data.get('current_price', 'N/A')}")
                    print(f"📈 Change: {stock_data.get('change_percent', 0):.2f}%")
                    print(f"\n🔍 Analysis: {analysis.analysis[:200]}...")
                    print(f"\n💡 Recommendation: {analysis.recommendation}")

                elif "comparison_analysis" in results:
                    comparison = results["comparison_analysis"]
                    print(f"\n📝 Comparison: {comparison.comparison_summary}")

                    # Parse rankings string to list
                    parsed_rankings = _parse_json_list(comparison.rankings)
                    print(f"\n🏆 Rankings: {', '.join(parsed_rankings[:2])}")

        except Exception as e:
            print(f"❌ Error: {e}")


async def main() -> None:
    await demo_financial_analysis()


if __name__ == "__main__":
    asyncio.run(main())
