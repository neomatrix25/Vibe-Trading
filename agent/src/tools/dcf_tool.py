"""DCF Valuation Tool — Discounted Cash Flow model.

Fetches fundamentals from ArcQuant (Finnhub), projects earnings,
discounts to present value, and estimates fair value per share.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from src.agent.tools import BaseTool

ARCQUANT_URL = os.environ.get("ARCQUANT_URL", "http://localhost:3003")


class DCFValuationTool(BaseTool):
    """Run a DCF (Discounted Cash Flow) valuation on a stock."""

    name = "dcf_valuation"
    description = (
        "Run a DCF valuation model for any US stock. "
        "Fetches real fundamentals (EPS, revenue, margins) from ArcQuant/Finnhub, "
        "projects earnings forward 5 years, discounts to present value, "
        "and compares fair value to current market price. "
        "Returns: fair value per share, upside/downside %, sensitivity table. "
        "Stocks only — not for crypto."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "US stock symbol (e.g. META, AAPL, MSFT)",
            },
            "growth_rate": {
                "type": "number",
                "description": "Annual earnings growth rate (default: estimated from data). E.g. 0.10 = 10%",
            },
            "discount_rate": {
                "type": "number",
                "description": "Discount rate / required return (default 0.10 = 10%)",
            },
            "terminal_multiple": {
                "type": "number",
                "description": "Terminal P/E multiple for year 5 (default 15)",
            },
            "projection_years": {
                "type": "integer",
                "description": "Years to project (default 5, max 10)",
            },
        },
        "required": ["symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        symbol = kwargs["symbol"].upper()

        # Fetch fundamentals from ArcQuant
        try:
            resp = requests.get(
                f"{ARCQUANT_URL}/api/internal/data",
                params={"type": "fundamentals", "symbol": symbol},
                timeout=15,
            )
            if resp.status_code != 200:
                return json.dumps({"error": f"Could not fetch fundamentals for {symbol}"})
            fundamentals = resp.json()
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch data: {e}"})

        # Also get current price
        try:
            price_resp = requests.get(
                f"{ARCQUANT_URL}/api/internal/data",
                params={"type": "price", "symbol": symbol},
                timeout=15,
            )
            price_data = price_resp.json() if price_resp.status_code == 200 else {}
        except Exception:
            price_data = {}

        current_price = price_data.get("price", 0)

        # Extract key metrics
        # Finnhub fundamentals can return different structures
        metric = fundamentals.get("metric", fundamentals)
        if isinstance(metric, list) and len(metric) > 0:
            metric = metric[0] if isinstance(metric[0], dict) else fundamentals

        eps = (
            metric.get("epsTTM")
            or metric.get("epsBasicExclExtraItemsTTM")
            or metric.get("epsNormalizedAnnual")
            or 0
        )
        pe = metric.get("peNormalizedAnnual") or metric.get("peTTM") or 0
        revenue_growth = metric.get("revenueGrowthQuarterlyYoy") or metric.get("revenueGrowth3Y") or 0
        net_margin = metric.get("netProfitMarginTTM") or metric.get("netMargin") or 0

        if eps <= 0:
            return json.dumps({
                "error": f"{symbol} has negative or zero EPS ({eps}). DCF requires positive earnings.",
                "eps": eps,
                "hint": "This stock may not be suitable for DCF — try a profitable company.",
            })

        # Parameters
        growth = kwargs.get("growth_rate") or min(max(revenue_growth / 100 if revenue_growth > 1 else revenue_growth, 0.03), 0.30)
        discount = kwargs.get("discount_rate", 0.10)
        terminal_pe = kwargs.get("terminal_multiple", 15)
        years = min(kwargs.get("projection_years", 5), 10)

        # Project EPS
        projections = []
        for yr in range(1, years + 1):
            projected_eps = eps * (1 + growth) ** yr
            pv = projected_eps / (1 + discount) ** yr
            projections.append({
                "year": yr,
                "eps": round(projected_eps, 2),
                "pv": round(pv, 2),
            })

        # Terminal value
        terminal_eps = eps * (1 + growth) ** years
        terminal_value = terminal_eps * terminal_pe
        pv_terminal = terminal_value / (1 + discount) ** years

        # Fair value
        pv_earnings = sum(p["pv"] for p in projections)
        fair_value = round(pv_earnings + pv_terminal, 2)

        # Upside/downside
        upside = round((fair_value / current_price - 1) * 100, 1) if current_price > 0 else 0
        verdict = "UNDERVALUED" if upside > 15 else "OVERVALUED" if upside < -15 else "FAIR VALUE"

        # Sensitivity table (3x3: growth × discount rate)
        sensitivity = []
        for g in [growth - 0.03, growth, growth + 0.03]:
            for d in [discount - 0.02, discount, discount + 0.02]:
                if g <= 0 or d <= 0:
                    continue
                tv_eps = eps * (1 + g) ** years
                tv = tv_eps * terminal_pe
                pv_e = sum(eps * (1 + g) ** yr / (1 + d) ** yr for yr in range(1, years + 1))
                pv_t = tv / (1 + d) ** years
                fv = round(pv_e + pv_t, 2)
                sensitivity.append({
                    "growth": f"{g*100:.1f}%",
                    "discount": f"{d*100:.1f}%",
                    "fair_value": fv,
                })

        return json.dumps({
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "fair_value": fair_value,
            "upside_pct": upside,
            "verdict": verdict,
            "inputs": {
                "eps_ttm": round(eps, 2),
                "growth_rate": f"{growth*100:.1f}%",
                "discount_rate": f"{discount*100:.1f}%",
                "terminal_pe": terminal_pe,
                "projection_years": years,
            },
            "fundamentals": {
                "pe_ratio": round(pe, 2) if pe else None,
                "revenue_growth": f"{revenue_growth:.1f}%" if revenue_growth else None,
                "net_margin": f"{net_margin:.1f}%" if net_margin else None,
            },
            "projections": projections,
            "terminal_value": {
                "terminal_eps": round(terminal_eps, 2),
                "terminal_value": round(terminal_value, 2),
                "pv_terminal": round(pv_terminal, 2),
            },
            "sensitivity": sensitivity,
        }, ensure_ascii=False)
