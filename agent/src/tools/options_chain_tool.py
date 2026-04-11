"""Options Chain tool — real market data via yfinance.

Returns real bid/ask, implied volatility, volume, open interest
for any US stock options. Free, no API key needed.

Data source: Yahoo Finance (via yfinance library).
"""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import BaseTool


class OptionsChainTool(BaseTool):
    """Get real options chain data for a US stock."""

    name = "options_chain"
    description = (
        "Get REAL options chain data (not theoretical) for any US stock. "
        "Returns: expiry dates, strikes, bid, ask, last price, implied volatility, "
        "volume, open interest, Greeks. Data from Yahoo Finance. "
        "Use this INSTEAD of options_pricing for real market prices. "
        "options_pricing is Black-Scholes math only. This gives actual market data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "US stock symbol (e.g. META, AAPL, TSLA, NVDA)",
            },
            "expiry": {
                "type": "string",
                "description": "Expiry date YYYY-MM-DD (optional — omit to get nearest expiry). Use get_expiries first to see available dates.",
            },
            "strike_range": {
                "type": "number",
                "description": "How far from ATM to show strikes in % (default 5 = ±5% from current price)",
            },
            "action": {
                "type": "string",
                "description": "Action: 'chain' (default — get calls+puts), 'expiries' (list available expiry dates)",
                "enum": ["chain", "expiries"],
            },
        },
        "required": ["symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        try:
            import yfinance as yf
        except ImportError:
            return json.dumps({"error": "yfinance not installed"})

        symbol = kwargs["symbol"].upper()
        action = kwargs.get("action", "chain")

        try:
            ticker = yf.Ticker(symbol)

            # Get available expiry dates
            expiries = ticker.options
            if not expiries:
                return json.dumps({
                    "error": f"No options data found for {symbol}. US stocks only.",
                })

            if action == "expiries":
                return json.dumps({
                    "symbol": symbol,
                    "expiry_dates": list(expiries),
                    "count": len(expiries),
                    "nearest": expiries[0],
                })

            # Get chain for specific expiry
            expiry = kwargs.get("expiry") or expiries[0]
            if expiry not in expiries:
                return json.dumps({
                    "error": f"Expiry {expiry} not available",
                    "available": list(expiries[:10]),
                })

            chain = ticker.option_chain(expiry)
            spot = ticker.info.get("currentPrice") or ticker.info.get("regularMarketPrice", 0)
            strike_pct = kwargs.get("strike_range", 5) / 100
            low = spot * (1 - strike_pct)
            high = spot * (1 + strike_pct)

            # Filter near ATM
            calls = chain.calls[
                (chain.calls.strike >= low) & (chain.calls.strike <= high)
            ]
            puts = chain.puts[
                (chain.puts.strike >= low) & (chain.puts.strike <= high)
            ]

            def format_row(row: Any) -> dict:
                return {
                    "strike": float(row.strike),
                    "lastPrice": float(row.lastPrice),
                    "bid": float(row.bid),
                    "ask": float(row.ask),
                    "iv": round(float(row.impliedVolatility), 4),
                    "volume": int(row.volume) if row.volume == row.volume else 0,
                    "openInterest": int(row.openInterest) if row.openInterest == row.openInterest else 0,
                    "inTheMoney": bool(row.inTheMoney),
                }

            result = {
                "symbol": symbol,
                "spot_price": round(spot, 2),
                "expiry": expiry,
                "days_to_expiry": _days_to(expiry),
                "calls": [format_row(r) for _, r in calls.iterrows()],
                "puts": [format_row(r) for _, r in puts.iterrows()],
                "call_count": len(calls),
                "put_count": len(puts),
            }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": str(e)})


def _days_to(date_str: str) -> int:
    """Days from today to a date string."""
    from datetime import datetime
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
        return max(0, (target - datetime.now()).days)
    except Exception:
        return 0
