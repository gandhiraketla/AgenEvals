import os
import math
import requests
import yfinance as yf
from langchain_core.tools import tool


@tool
def stock_price_fetcher(ticker: str) -> dict:
    """
    Fetch the current stock price, daily change, 52-week high/low, and market cap
    for a given stock ticker symbol (e.g. AAPL, TSLA, MSFT, NVDA).
    Use this tool whenever the user asks about a stock's current price, performance,
    recent movement, or valuation metrics.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        hist = stock.history(period="5d")

        if hist.empty:
            return {"error": f"No data found for ticker '{ticker}'"}

        current_price = round(hist["Close"].iloc[-1], 2)
        prev_price = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else current_price
        change = round(current_price - prev_price, 2)
        change_pct = round((change / prev_price) * 100, 2) if prev_price else 0

        return {
            "ticker": ticker.upper(),
            "current_price": current_price,
            "previous_close": prev_price,
            "change": change,
            "change_pct": f"{change_pct}%",
            "52w_high": round(info.get("fiftyTwoWeekHigh", 0), 2),
            "52w_low": round(info.get("fiftyTwoWeekLow", 0), 2),
            "market_cap": info.get("marketCap", "N/A"),
            "volume": info.get("volume", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def news_fetcher(ticker: str, limit: int = 5) -> list[dict]:
    """
    Fetch recent financial news headlines and summaries for a given stock ticker symbol
    (e.g. AAPL, TSLA, MSFT, NVDA). Use this tool when the user asks why a stock moved,
    wants recent news, sentiment analysis, or asks whether they should buy/sell based on news.
    The input must be a stock ticker symbol, not a general search query.
    """
    import datetime

    api_key = os.getenv("FINHUB_API_KEY")
    if not api_key:
        return [{"error": "FINHUB_API_KEY not set in environment"}]

    to_date = datetime.date.today().isoformat()
    from_date = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker.upper(),
        "from": from_date,
        "to": to_date,
        "token": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json()
        return [
            {
                "title": a.get("headline"),
                "source": a.get("source"),
                "published_at": datetime.datetime.fromtimestamp(a.get("datetime", 0)).isoformat(),
                "summary": a.get("summary"),
                "sentiment": a.get("sentiment"),
                "url": a.get("url"),
            }
            for a in articles[:limit]
        ]
    except Exception as e:
        return [{"error": str(e)}]


@tool
def calculator(expression: str) -> dict:
    """
    Perform financial calculations such as ROI, percentage change, profit/loss,
    portfolio value, or any arithmetic expression. Use this tool whenever the user
    asks for a numerical computation, percentage, return on investment, or comparison
    between prices. Input should be a valid Python math expression as a string.
    Example: '((210 - 150) / 150) * 100' to compute ROI.
    """
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return {"expression": expression, "result": round(float(result), 4)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


@tool
def portfolio_analyzer(tickers: list[str]) -> dict:
    """
    Analyze a portfolio of multiple stock tickers. Returns current prices, individual
    weights if equal-weighted, total portfolio value per unit, best and worst performers,
    and a diversification summary. Use this tool when the user asks to analyze, review,
    or rebalance a portfolio of multiple stocks.
    """
    if not tickers:
        return {"error": "No tickers provided"}

    results = {}
    errors = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker.upper())
            hist = stock.history(period="5d")
            if hist.empty:
                errors.append(f"No data for {ticker}")
                continue
            current = round(hist["Close"].iloc[-1], 2)
            prev = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else current
            change_pct = round(((current - prev) / prev) * 100, 2) if prev else 0
            results[ticker.upper()] = {"price": current, "daily_change_pct": change_pct}
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")

    if not results:
        return {"error": "Could not fetch data for any ticker", "details": errors}

    total_value = sum(v["price"] for v in results.values())
    weight = round(100 / len(results), 2)
    best = max(results, key=lambda t: results[t]["daily_change_pct"])
    worst = min(results, key=lambda t: results[t]["daily_change_pct"])

    return {
        "portfolio": results,
        "total_equal_weighted_value": round(total_value, 2),
        "equal_weight_pct": f"{weight}% each",
        "best_performer_today": best,
        "worst_performer_today": worst,
        "num_holdings": len(results),
        "errors": errors if errors else None,
    }
