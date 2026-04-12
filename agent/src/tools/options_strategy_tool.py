"""Options Strategy P&L Calculator — multi-leg payoff analysis.

Computes max profit, max loss, breakeven points, and payoff curve
for any combination of calls and puts (iron condor, straddle, spread, etc.).

Uses real market prices from yfinance for premium costs.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

from src.agent.tools import BaseTool


class OptionsStrategyTool(BaseTool):
    """Calculate P&L for multi-leg options strategies."""

    name = "options_strategy"
    description = (
        "Calculate payoff for any options strategy (iron condor, straddle, spread, etc.). "
        "Define legs with type (call/put), strike, action (buy/sell), premium. "
        "Returns: max profit, max loss, breakeven points, P&L at key prices. "
        "Use options_chain first to get real bid/ask prices for the legs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "legs": {
                "type": "array",
                "description": "Array of option legs: [{type:'call'|'put', strike:number, action:'buy'|'sell', premium:number, qty:number}]",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["call", "put"]},
                        "strike": {"type": "number"},
                        "action": {"type": "string", "enum": ["buy", "sell"]},
                        "premium": {"type": "number", "description": "Option premium per share"},
                        "qty": {"type": "integer", "description": "Number of contracts (default 1)"},
                    },
                    "required": ["type", "strike", "action", "premium"],
                },
            },
            "spot": {
                "type": "number",
                "description": "Current stock price (for reference)",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol (for labeling only)",
            },
        },
        "required": ["legs"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        legs = kwargs.get("legs", [])
        if not legs:
            return json.dumps({"error": "At least one leg required"})

        spot = kwargs.get("spot", 0)
        symbol = kwargs.get("symbol", "")

        # Parse legs
        parsed_legs: List[Dict] = []
        net_premium = 0.0

        for i, leg in enumerate(legs):
            opt_type = leg.get("type", "call")
            strike = float(leg.get("strike", 0))
            action = leg.get("action", "buy")
            premium = float(leg.get("premium", 0))
            qty = int(leg.get("qty", 1))
            multiplier = 100  # standard US equity options

            if strike <= 0:
                return json.dumps({"error": f"Leg {i+1}: strike must be positive"})

            sign = 1 if action == "buy" else -1
            cost = sign * premium * qty * multiplier
            net_premium += cost

            parsed_legs.append({
                "type": opt_type,
                "strike": strike,
                "action": action,
                "premium": premium,
                "qty": qty,
                "sign": sign,
                "multiplier": multiplier,
            })

        # Compute payoff at a range of prices
        all_strikes = [leg["strike"] for leg in parsed_legs]
        min_strike = min(all_strikes)
        max_strike = max(all_strikes)
        margin = (max_strike - min_strike) * 0.5 or max_strike * 0.1
        price_low = max(0, min_strike - margin)
        price_high = max_strike + margin

        # Generate price points
        step = (price_high - price_low) / 50
        prices = [round(price_low + i * step, 2) for i in range(51)]
        # Also include exact strikes
        for s in all_strikes:
            if s not in prices:
                prices.append(s)
        if spot > 0 and spot not in prices:
            prices.append(spot)
        prices.sort()

        # Compute payoff at each price
        payoffs: List[Dict] = []
        max_profit = -float("inf")
        max_loss = float("inf")
        breakevens: List[float] = []

        prev_pnl = None
        for price in prices:
            total_pnl = 0.0
            for leg in parsed_legs:
                if leg["type"] == "call":
                    intrinsic = max(0, price - leg["strike"])
                else:
                    intrinsic = max(0, leg["strike"] - price)

                leg_pnl = leg["sign"] * (intrinsic - leg["premium"]) * leg["qty"] * leg["multiplier"]
                total_pnl += leg_pnl

            total_pnl = round(total_pnl, 2)
            payoffs.append({"price": price, "pnl": total_pnl})

            if total_pnl > max_profit:
                max_profit = total_pnl
            if total_pnl < max_loss:
                max_loss = total_pnl

            # Find breakevens (where PnL crosses zero)
            if prev_pnl is not None and prev_pnl * total_pnl < 0:
                # Linear interpolation
                prev_price = prices[prices.index(price) - 1]
                be = prev_price + (0 - prev_pnl) / (total_pnl - prev_pnl) * (price - prev_price)
                breakevens.append(round(be, 2))
            prev_pnl = total_pnl

        # Strategy description
        strategy_name = _identify_strategy(parsed_legs)

        return json.dumps({
            "symbol": symbol,
            "spot": spot,
            "strategy": strategy_name,
            "legs": [{
                "type": l["type"],
                "strike": l["strike"],
                "action": l["action"],
                "premium": l["premium"],
                "qty": l["qty"],
            } for l in parsed_legs],
            "net_premium": round(net_premium, 2),
            "max_profit": round(max_profit, 2) if max_profit != float("inf") else "unlimited",
            "max_loss": round(max_loss, 2) if max_loss != -float("inf") else "unlimited",
            "breakevens": breakevens,
            "risk_reward": round(abs(max_profit / max_loss), 2) if max_loss != 0 and isinstance(max_profit, (int, float)) and isinstance(max_loss, (int, float)) else "N/A",
            "payoff_at_strikes": [p for p in payoffs if p["price"] in all_strikes or (spot > 0 and abs(p["price"] - spot) < step)],
        }, ensure_ascii=False)


def _identify_strategy(legs: List[Dict]) -> str:
    """Try to identify common strategy names."""
    n = len(legs)
    types = [l["type"] for l in legs]
    actions = [l["action"] for l in legs]
    strikes = [l["strike"] for l in legs]

    if n == 1:
        return f"{'Long' if actions[0] == 'buy' else 'Short'} {types[0].title()}"
    if n == 2:
        if types[0] == types[1] == "call" and "buy" in actions and "sell" in actions:
            return "Bull Call Spread" if strikes[actions.index("buy")] < strikes[actions.index("sell")] else "Bear Call Spread"
        if types[0] == types[1] == "put" and "buy" in actions and "sell" in actions:
            return "Bear Put Spread" if strikes[actions.index("buy")] > strikes[actions.index("sell")] else "Bull Put Spread"
        if set(types) == {"call", "put"} and all(a == "buy" for a in actions):
            return "Long Straddle" if strikes[0] == strikes[1] else "Long Strangle"
        if set(types) == {"call", "put"} and all(a == "sell" for a in actions):
            return "Short Straddle" if strikes[0] == strikes[1] else "Short Strangle"
    if n == 4:
        if types.count("call") == 2 and types.count("put") == 2:
            if actions.count("buy") == 2 and actions.count("sell") == 2:
                return "Iron Condor"
    return f"{n}-Leg Strategy"
