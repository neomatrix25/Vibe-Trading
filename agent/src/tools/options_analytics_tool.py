"""Options Analytics — max pain, expected move, IV surface.

Three computations in one tool to keep tool count low.
All use real data from yfinance options chains.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

from src.agent.tools import BaseTool


def _is_weekday(date_str: str) -> bool:
    """Check if date is Mon-Fri."""
    try:
        from datetime import datetime
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() < 5
    except Exception:
        return False


def _filter_valid_expiries(expiries: tuple) -> list:
    """Filter to weekday expiries only."""
    valid = [d for d in expiries if _is_weekday(d)]
    return valid if valid else list(expiries)


class OptionsAnalyticsTool(BaseTool):
    """Compute max pain, expected move, or IV surface for US stock options."""

    name = "options_analytics"
    description = (
        "Options analytics using REAL market data. Three modes:\n"
        "- action='max_pain': Find strike where most options expire worthless\n"
        "- action='expected_move': Market-implied price range from ATM straddle\n"
        "- action='iv_surface': Implied volatility across strikes and expiries (vol smile/skew)"
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "US stock symbol (e.g. META, AAPL, TSLA)",
            },
            "action": {
                "type": "string",
                "description": "max_pain | expected_move | iv_surface",
                "enum": ["max_pain", "expected_move", "iv_surface"],
            },
            "expiry": {
                "type": "string",
                "description": "Expiry date YYYY-MM-DD. Omit for nearest.",
            },
            "num_expiries": {
                "type": "integer",
                "description": "For iv_surface: how many expiries to include (default 3, max 5)",
            },
        },
        "required": ["symbol", "action"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        try:
            import yfinance as yf
        except ImportError:
            return json.dumps({"error": "yfinance not installed"})

        symbol = kwargs["symbol"].upper()
        action = kwargs["action"]

        try:
            ticker = yf.Ticker(symbol)
            expiries = ticker.options
            if not expiries:
                return json.dumps({"error": f"No options data for {symbol}"})

            # Get spot price
            info = ticker.info or {}
            spot = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0
            if spot == 0:
                try:
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        spot = float(hist["Close"].iloc[-1])
                except Exception:
                    pass

            valid_expiries = _filter_valid_expiries(expiries)
            expiry = kwargs.get("expiry") or valid_expiries[0]
            if expiry not in expiries:
                return json.dumps({"error": f"Expiry {expiry} not available", "available": list(expiries[:10])})

            if action == "max_pain":
                return self._max_pain(ticker, symbol, spot, expiry)
            elif action == "expected_move":
                return self._expected_move(ticker, symbol, spot, expiry)
            elif action == "iv_surface":
                num = min(kwargs.get("num_expiries", 3), 5)
                return self._iv_surface(ticker, symbol, spot, valid_expiries[:num])
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _max_pain(self, ticker: Any, symbol: str, spot: float, expiry: str) -> str:
        """Find strike where total option holder pain is maximized (= most expire worthless)."""
        chain = ticker.option_chain(expiry)

        # Collect all strikes with OI
        call_data: Dict[float, int] = {}
        put_data: Dict[float, int] = {}

        for _, row in chain.calls.iterrows():
            oi = int(row.openInterest) if not math.isnan(row.openInterest) else 0
            if oi > 0:
                call_data[float(row.strike)] = oi

        for _, row in chain.puts.iterrows():
            oi = int(row.openInterest) if not math.isnan(row.openInterest) else 0
            if oi > 0:
                put_data[float(row.strike)] = oi

        all_strikes = sorted(set(list(call_data.keys()) + list(put_data.keys())))
        if not all_strikes:
            return json.dumps({"error": "No open interest data"})

        # For each test price, compute total intrinsic value of all options
        pain: Dict[float, float] = {}
        for test_price in all_strikes:
            call_pain = sum(max(0, test_price - k) * oi for k, oi in call_data.items())
            put_pain = sum(max(0, k - test_price) * oi for k, oi in put_data.items())
            pain[test_price] = call_pain + put_pain

        max_pain_strike = min(pain, key=pain.get)  # type: ignore
        total_call_oi = sum(call_data.values())
        total_put_oi = sum(put_data.values())

        return json.dumps({
            "symbol": symbol,
            "expiry": expiry,
            "spot": round(spot, 2),
            "max_pain_strike": max_pain_strike,
            "distance_from_spot": round(max_pain_strike - spot, 2),
            "distance_pct": round((max_pain_strike - spot) / spot * 100, 2),
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_oi_ratio": round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0,
            "pain_by_strike": {str(k): round(v) for k, v in sorted(pain.items()) if abs(k - spot) / spot < 0.1},
        }, ensure_ascii=False)

    def _expected_move(self, ticker: Any, symbol: str, spot: float, expiry: str) -> str:
        """Calculate expected move from ATM straddle price."""
        chain = ticker.option_chain(expiry)

        # Find ATM strike (closest to spot)
        all_strikes = sorted(chain.calls.strike.unique())
        atm_strike = min(all_strikes, key=lambda k: abs(k - spot))

        # Get ATM call and put
        atm_call = chain.calls[chain.calls.strike == atm_strike]
        atm_put = chain.puts[chain.puts.strike == atm_strike]

        if atm_call.empty or atm_put.empty:
            return json.dumps({"error": "Could not find ATM options"})

        call_mid = (float(atm_call.bid.iloc[0]) + float(atm_call.ask.iloc[0])) / 2
        put_mid = (float(atm_put.bid.iloc[0]) + float(atm_put.ask.iloc[0])) / 2
        straddle_price = call_mid + put_mid

        call_iv = float(atm_call.impliedVolatility.iloc[0]) if not math.isnan(atm_call.impliedVolatility.iloc[0]) else 0
        put_iv = float(atm_put.impliedVolatility.iloc[0]) if not math.isnan(atm_put.impliedVolatility.iloc[0]) else 0
        avg_iv = (call_iv + put_iv) / 2

        return json.dumps({
            "symbol": symbol,
            "expiry": expiry,
            "spot": round(spot, 2),
            "atm_strike": atm_strike,
            "atm_call_mid": round(call_mid, 2),
            "atm_put_mid": round(put_mid, 2),
            "straddle_price": round(straddle_price, 2),
            "expected_move_dollars": round(straddle_price, 2),
            "expected_move_pct": round(straddle_price / spot * 100, 2),
            "expected_range": {
                "low": round(spot - straddle_price, 2),
                "high": round(spot + straddle_price, 2),
            },
            "atm_iv": round(avg_iv, 4),
        }, ensure_ascii=False)

    def _iv_surface(self, ticker: Any, symbol: str, spot: float, expiries: List[str]) -> str:
        """IV across strikes and expiries — shows vol smile/skew."""
        surface: List[Dict] = []

        for expiry in expiries:
            try:
                chain = ticker.option_chain(expiry)
                # Get calls near ATM (±15%)
                low = spot * 0.85
                high = spot * 1.15
                calls = chain.calls[(chain.calls.strike >= low) & (chain.calls.strike <= high)]

                strikes_data = []
                for _, row in calls.iterrows():
                    iv = float(row.impliedVolatility) if not math.isnan(row.impliedVolatility) else 0
                    if iv > 0:
                        moneyness = round((float(row.strike) / spot - 1) * 100, 1)
                        strikes_data.append({
                            "strike": float(row.strike),
                            "moneyness_pct": moneyness,
                            "iv": round(iv, 4),
                            "oi": int(row.openInterest) if not math.isnan(row.openInterest) else 0,
                        })

                if strikes_data:
                    surface.append({
                        "expiry": expiry,
                        "strikes": strikes_data,
                    })
            except Exception:
                continue

        if not surface:
            return json.dumps({"error": "Could not build IV surface"})

        # Find ATM IV for each expiry
        atm_ivs = []
        for exp_data in surface:
            atm = min(exp_data["strikes"], key=lambda s: abs(s["moneyness_pct"]))
            atm_ivs.append({"expiry": exp_data["expiry"], "atm_iv": atm["iv"]})

        return json.dumps({
            "symbol": symbol,
            "spot": round(spot, 2),
            "surface": surface,
            "atm_term_structure": atm_ivs,
            "skew_note": "If OTM puts have higher IV than OTM calls = negative skew (fear premium). Normal for equities.",
        }, ensure_ascii=False)
