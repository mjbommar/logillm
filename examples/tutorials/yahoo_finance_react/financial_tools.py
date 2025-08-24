"""Financial data tools for Yahoo Finance ReAct tutorial."""

from datetime import datetime, timedelta
from typing import Any

import yfinance as yf


def get_stock_price(symbol: str) -> dict[str, Any]:
    """Get current stock price and basic information."""
    try:
        stock = yf.Ticker(symbol.upper())
        info = stock.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        previous_close = info.get("previousClose")

        if current_price is None:
            history = stock.history(period="1d")
            if not history.empty:
                current_price = float(history["Close"].iloc[-1])
                previous_close = (
                    float(history["Close"].iloc[-2]) if len(history) > 1 else current_price
                )

        change = current_price - previous_close if (current_price and previous_close) else 0
        change_percent = (change / previous_close * 100) if previous_close else 0

        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "company_name": info.get("longName", symbol.upper()),
            "market_cap": info.get("marketCap"),
            "volume": info.get("volume"),
            "status": "success",
        }

    except Exception as e:
        return {"symbol": symbol.upper(), "status": "error", "error": str(e)}


def compare_stocks(symbols: list[str]) -> dict[str, Any]:
    """Compare multiple stocks side by side."""
    results = {}
    for symbol in symbols:
        results[symbol] = get_stock_price(symbol)

    successful_stocks = {k: v for k, v in results.items() if v["status"] == "success"}

    if len(successful_stocks) > 1:
        best_performer = max(
            successful_stocks.keys(),
            key=lambda x: successful_stocks[x].get("change_percent", -float("inf")),
        )
        worst_performer = min(
            successful_stocks.keys(),
            key=lambda x: successful_stocks[x].get("change_percent", float("inf")),
        )

        return {
            "comparison": results,
            "best_performer": best_performer,
            "worst_performer": worst_performer,
            "comparison_summary": f"{best_performer} is the best performer, {worst_performer} is the worst",
        }

    return {"comparison": results}


def get_stock_news(symbol: str, max_articles: int = 3) -> dict[str, Any]:
    """Get recent news for a stock (mock implementation)."""
    mock_news = [
        {
            "title": f"{symbol} Reports Strong Q4 Earnings",
            "summary": f"{symbol} exceeded analyst expectations with strong revenue growth.",
            "source": "Financial Times",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "title": f"Analysts Upgrade {symbol} Rating",
            "summary": f'Major investment firm upgrades {symbol} to "Buy" rating.',
            "source": "Bloomberg",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
        },
    ]

    return {"symbol": symbol.upper(), "news": mock_news[:max_articles], "status": "success"}
