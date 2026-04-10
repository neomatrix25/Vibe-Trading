"""ArcQuant ACP Tool — lets the agent call ArcQuant's data endpoints.

Provides access to:
  - Real-time prices (Binance crypto, Finnhub stocks)
  - 150+ technical indicators
  - Company fundamentals, insider activity, earnings
  - Strategy signals and backtests
  - DeFi yield scoring
  - Signal engines (technical vote, candlestick, ichimoku, etc.)

All calls go to ArcQuant at localhost:3003 (internal, no USDC needed).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests

from src.agent.tools import BaseTool

ARCQUANT_URL = os.environ.get("ARCQUANT_URL", "http://localhost:3003")
TIMEOUT = 15  # seconds


def _call(endpoint: str, params: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Call ArcQuant API endpoint."""
    url = f"{ARCQUANT_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def _internal(data_type: str, params: Dict[str, str]) -> Dict[str, Any]:
    """Call internal data endpoint (no ACP payment needed)."""
    params["type"] = data_type
    return _call("/api/internal/data", params)


class ArcQuantPriceTool(BaseTool):
    """Get real-time price for a symbol (crypto or stocks)."""

    name = "arcquant_price"
    description = (
        "Get real-time price, 24h change, and volume for any symbol. "
        "Crypto (BTCUSDT, ETHUSDT) via Binance. Stocks (AAPL, MSFT) via Finnhub."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol like BTCUSDT, AAPL, ETHUSDT, MSFT",
            },
        },
        "required": ["symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        symbol = kwargs["symbol"].upper()
        data = _internal("price", {"symbol": symbol})
        return json.dumps(data, ensure_ascii=False)


class ArcQuantIndicatorsTool(BaseTool):
    """Compute technical indicators on any symbol."""

    name = "arcquant_indicators"
    description = (
        "Compute technical indicators: RSI, MACD, EMA, SMA, Bollinger Bands, "
        "ATR, VWAP, Stochastic, ADX, OBV, and 140+ more. Returns current value + last 20."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol like BTCUSDT or AAPL",
            },
            "indicator": {
                "type": "string",
                "description": "Indicator name: RSI, MACD, EMA, SMA, BOLLINGER, ATR, VWAP, STOCHASTIC, ADX, OBV",
            },
            "period": {
                "type": "integer",
                "description": "Indicator period (default 14)",
            },
        },
        "required": ["symbol", "indicator"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        params = {
            "symbol": kwargs["symbol"].upper(),
            "indicator": kwargs["indicator"].upper(),
        }
        if "period" in kwargs:
            params["period"] = str(kwargs["period"])
        data = _internal("indicators", params)
        return json.dumps(data, ensure_ascii=False)


class ArcQuantSignalEngineTool(BaseTool):
    """Run one of 8 signal engines on a symbol."""

    name = "arcquant_signal_engine"
    description = (
        "Run a signal engine on a symbol. Available engines: "
        "technical-vote (EMA+ADX+BB+RSI+OBV voting), "
        "candlestick (10 patterns), ichimoku (TK cross + cloud), "
        "vol-regime (HV percentile), smc (Smart Money Concepts), "
        "seasonal (month/weekday effects), pairs (Z-score mean reversion), "
        "multi-factor (cross-sectional ranking). Returns direction + confidence."
    )
    parameters = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "description": "Engine type: technical-vote, candlestick, ichimoku, vol-regime, smc, seasonal",
                "enum": ["technical-vote", "candlestick", "ichimoku", "vol-regime", "smc", "seasonal"],
            },
            "symbol": {
                "type": "string",
                "description": "Symbol like BTCUSDT or AAPL",
            },
        },
        "required": ["type", "symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        params = {
            "type": kwargs["type"],
            "symbol": kwargs["symbol"].upper(),
        }
        data = _call("/api/signals/engines", params)
        return json.dumps(data, ensure_ascii=False)


class ArcQuantFundamentalsTool(BaseTool):
    """Get company fundamentals (stocks only)."""

    name = "arcquant_fundamentals"
    description = (
        "Get company fundamentals: P/E, revenue, margins, EPS, market cap. "
        "Stocks only (AAPL, MSFT, GOOGL, etc.). Via Finnhub."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock symbol like AAPL, MSFT",
            },
        },
        "required": ["symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        data = _internal("fundamentals", {"symbol": kwargs["symbol"].upper()})
        return json.dumps(data, ensure_ascii=False)


class ArcQuantSchemaDiscoveryTool(BaseTool):
    """Discover ArcQuant's available indicators, node types, and strategy format."""

    name = "arcquant_schema"
    description = (
        "Get ArcQuant's full schema: all available indicators, DAG node types, "
        "filter operators, connection rules, and strategy format. "
        "Use this to understand what's available before building strategies."
    )
    parameters = {
        "type": "object",
        "properties": {},
    }
    repeatable = False

    def execute(self, **kwargs: Any) -> str:
        data = _call("/api/v1/schema")
        result = json.dumps(data, ensure_ascii=False)
        if len(result) > 8000:
            result = result[:8000] + "\n... [truncated]"
        return result


class ArcQuantInsiderTool(BaseTool):
    """Get insider trading activity for a stock."""

    name = "arcquant_insider"
    description = (
        "Get recent insider buys and sells with transaction details. "
        "Stocks only (AAPL, MSFT, TSLA, etc.). Shows who bought/sold, "
        "how many shares, at what price, and when."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock symbol like AAPL, MSFT, TSLA",
            },
        },
        "required": ["symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        data = _internal("insider", {"symbol": kwargs["symbol"].upper()})
        return json.dumps(data, ensure_ascii=False)


class ArcQuantEarningsTool(BaseTool):
    """Get earnings history and estimates for a stock."""

    name = "arcquant_earnings"
    description = (
        "Get earnings history: actual EPS vs estimate, surprise %, "
        "and upcoming earnings dates. Stocks only (AAPL, MSFT, etc.)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock symbol like AAPL, MSFT",
            },
        },
        "required": ["symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        data = _internal("earnings", {"symbol": kwargs["symbol"].upper()})
        return json.dumps(data, ensure_ascii=False)


class ArcQuantChartTool(BaseTool):
    """Generate a price chart with indicators and signals."""

    name = "arcquant_chart"
    description = (
        "Generate a price chart PNG with technical indicators overlaid. "
        "Supports: SMA, EMA, Bollinger Bands, RSI, MACD. "
        "Returns base64-encoded PNG image data. "
        "Works for both crypto (BTCUSDT) and stocks (AAPL)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol like BTCUSDT or AAPL",
            },
            "period": {
                "type": "string",
                "description": "Chart period: 1d, 1w, 1m, 3m, 6m, 1y (default 1m)",
            },
            "indicators": {
                "type": "string",
                "description": "Comma-separated indicators: sma,ema,bollinger,rsi,macd",
            },
        },
        "required": ["symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        params: Dict[str, str] = {"symbol": kwargs["symbol"].upper()}
        if "period" in kwargs:
            params["period"] = kwargs["period"]
        if "indicators" in kwargs:
            params["indicators"] = kwargs["indicators"]
        data = _internal("chart", params)
        return json.dumps(data, ensure_ascii=False)
