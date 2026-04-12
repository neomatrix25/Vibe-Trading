"""Options Chain tool — real market data via yfinance (production quality).

Returns real bid/ask, implied volatility, volume, open interest,
plus computed Greeks (delta, gamma, theta, vega) per strike.

Data source: Yahoo Finance (yfinance library, free, no API key).
Greeks: Black-Scholes computation from market IV.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.agent.tools import BaseTool

# Risk-free rate (US 10Y treasury approx)
RISK_FREE_RATE = 0.043


def _is_friday(date_str: str) -> bool:
    """Check if a date is a Friday. Standard options expiry day."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() == 4
    except Exception:
        return False


def _trading_days_to(date_str: str) -> int:
    """Business days from today to target date."""
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
        now = datetime.now()
        if target <= now:
            return 0
        days = 0
        current = now
        while current < target:
            current += timedelta(days=1)
            if current.weekday() < 5:
                days += 1
        return days
    except Exception:
        return 0


def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "call") -> Dict[str, float]:
    """Black-Scholes Greeks from spot, strike, time (years), rate, IV."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    try:
        from scipy.stats import norm
        import numpy as np

        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        nd1 = norm.pdf(d1)
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)

        if opt_type == "call":
            delta = round(Nd1, 4)
            theta_raw = (-S * nd1 * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * Nd2
        else:
            delta = round(Nd1 - 1, 4)
            theta_raw = (-S * nd1 * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)

        gamma = round(nd1 / (S * sigma * np.sqrt(T)), 6)
        theta = round(theta_raw / 365, 4)  # daily theta
        vega = round(S * nd1 * np.sqrt(T) / 100, 4)  # per 1% IV change

        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
    except Exception:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}


class OptionsChainTool(BaseTool):
    """Get real options chain data with Greeks for any US stock."""

    name = "options_chain"
    description = (
        "Get REAL options chain data for any US stock. Returns: expiry dates, "
        "strikes with bid/ask/last price, implied volatility, volume, open interest, "
        "and computed Greeks (delta, gamma, theta, vega). "
        "Use action='expiries' to list dates first, then action='chain' for data."
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
                "description": "Expiry date YYYY-MM-DD. Omit for nearest valid expiry.",
            },
            "strike_range": {
                "type": "number",
                "description": "How far from ATM in % (default 5 = ±5%). Use 10 for wider view.",
            },
            "action": {
                "type": "string",
                "description": "'chain' (default) or 'expiries' (list available dates)",
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
            expiries = ticker.options

            if not expiries:
                return json.dumps({"error": f"No options data for {symbol}. US stocks only."})

            # Filter to Friday expiries only (standard weekly/monthly)
            # Skips 0DTE Mon/Wed/daily options — not useful for analysis
            valid_expiries = [d for d in expiries if _is_friday(d)]
            if not valid_expiries:
                valid_expiries = list(expiries)  # fallback if no Fridays found

            if action == "expiries":
                return json.dumps({
                    "symbol": symbol,
                    "expiry_dates": valid_expiries,
                    "count": len(valid_expiries),
                    "nearest": valid_expiries[0],
                })

            # Pick expiry
            expiry = kwargs.get("expiry")
            if expiry:
                if expiry not in expiries:
                    return json.dumps({
                        "error": f"Expiry {expiry} not available",
                        "available": valid_expiries[:10],
                    })
            else:
                expiry = valid_expiries[0]

            # Get spot price with fallbacks
            info = ticker.info or {}
            spot = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
                or 0
            )
            if spot == 0:
                # Last resort: get from history
                try:
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        spot = float(hist["Close"].iloc[-1])
                except Exception:
                    pass

            if spot == 0:
                return json.dumps({"error": f"Could not get current price for {symbol}"})

            # Get chain
            chain = ticker.option_chain(expiry)
            strike_pct = kwargs.get("strike_range", 5) / 100
            low = spot * (1 - strike_pct)
            high = spot * (1 + strike_pct)

            days = _trading_days_to(expiry)
            T = max(days / 252, 1 / 252)  # time in years, min 1 trading day

            def format_row(row: Any, opt_type: str) -> dict:
                strike = float(row.strike)
                last = float(row.lastPrice) if not math.isnan(row.lastPrice) else 0
                bid = float(row.bid) if not math.isnan(row.bid) else 0
                ask = float(row.ask) if not math.isnan(row.ask) else 0
                iv = float(row.impliedVolatility) if not math.isnan(row.impliedVolatility) else 0
                vol = int(row.volume) if not math.isnan(row.volume) else 0
                oi = int(row.openInterest) if not math.isnan(row.openInterest) else 0
                itm = bool(row.inTheMoney) if hasattr(row, "inTheMoney") else False

                # Compute Greeks from market IV
                greeks = _bs_greeks(spot, strike, T, RISK_FREE_RATE, iv, opt_type)

                return {
                    "strike": strike,
                    "last": last,
                    "bid": bid,
                    "ask": ask,
                    "mid": round((bid + ask) / 2, 2) if bid + ask > 0 else last,
                    "iv": round(iv, 4),
                    "volume": vol,
                    "oi": oi,
                    "itm": itm,
                    "greeks": greeks,
                }

            # Filter calls and puts near ATM
            calls_df = chain.calls[(chain.calls.strike >= low) & (chain.calls.strike <= high)]
            puts_df = chain.puts[(chain.puts.strike >= low) & (chain.puts.strike <= high)]

            calls = [format_row(r, "call") for _, r in calls_df.iterrows()]
            puts = [format_row(r, "put") for _, r in puts_df.iterrows()]

            return json.dumps({
                "symbol": symbol,
                "spot": round(spot, 2),
                "expiry": expiry,
                "trading_days_to_expiry": days,
                "calls": calls,
                "puts": puts,
                "call_count": len(calls),
                "put_count": len(puts),
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": str(e)})
