"""Options Overview — one call, full options intelligence.

Combines: IV rank, put/call ratio, max pain, expected move,
unusual activity, top flow, gamma exposure, nearest expiry stats.

All from real yfinance data. One tool call = complete picture.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from src.agent.tools import BaseTool

# Risk-free rate approximation
RISK_FREE = 0.043


def _is_friday(d: str) -> bool:
    from datetime import datetime
    try:
        return datetime.strptime(d, "%Y-%m-%d").weekday() == 4
    except Exception:
        return False


def _bs_greeks_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """Compute gamma only (for GEX calculation)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0
    try:
        from scipy.stats import norm
        import numpy as np
        d1 = (np.log(S / K) + (RISK_FREE + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    except Exception:
        return 0


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float, handling NaN."""
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        f = float(val)
        return default if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return default


class OptionsOverviewTool(BaseTool):
    """Get complete options intelligence for any US stock in one call."""

    name = "options_overview"
    description = (
        "Complete options dashboard for any US stock. One call returns: "
        "IV rank, put/call ratio, max pain, expected move, unusual activity "
        "(volume >> OI), top flow (biggest $ trades), gamma exposure per strike, "
        "and nearest expiry stats. All from real market data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "US stock symbol (e.g. META, AAPL, TSLA)",
            },
        },
        "required": ["symbol"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        try:
            import yfinance as yf
            import numpy as np
        except ImportError:
            return json.dumps({"error": "yfinance or numpy not installed"})

        symbol = kwargs["symbol"].upper()

        try:
            ticker = yf.Ticker(symbol)
            expiries = ticker.options
            if not expiries:
                return json.dumps({"error": f"No options data for {symbol}. US stocks only."})

            # Spot price
            info = ticker.info or {}
            spot = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
                or 0
            )
            if spot == 0:
                try:
                    h = ticker.history(period="1d")
                    if not h.empty:
                        spot = float(h["Close"].iloc[-1])
                except Exception:
                    pass
            if spot == 0:
                return json.dumps({"error": f"Cannot get price for {symbol}"})

            # Pick nearest Friday expiry
            friday_expiries = [d for d in expiries if _is_friday(d)]
            if not friday_expiries:
                friday_expiries = list(expiries)
            nearest_expiry = friday_expiries[0]

            # Get chain
            chain = ticker.option_chain(nearest_expiry)

            # ── Days to expiry ──
            from datetime import datetime
            dte = max(1, (datetime.strptime(nearest_expiry, "%Y-%m-%d") - datetime.now()).days)
            T = dte / 365

            # ── Put/Call ratio ──
            total_call_oi = sum(_safe_int(r.openInterest) for _, r in chain.calls.iterrows())
            total_put_oi = sum(_safe_int(r.openInterest) for _, r in chain.puts.iterrows())
            pc_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0

            # ── Max pain ──
            call_oi_map: Dict[float, int] = {}
            put_oi_map: Dict[float, int] = {}
            for _, r in chain.calls.iterrows():
                oi = _safe_int(r.openInterest)
                if oi > 0:
                    call_oi_map[float(r.strike)] = oi
            for _, r in chain.puts.iterrows():
                oi = _safe_int(r.openInterest)
                if oi > 0:
                    put_oi_map[float(r.strike)] = oi

            all_strikes = sorted(set(list(call_oi_map.keys()) + list(put_oi_map.keys())))
            if all_strikes:
                pain = {}
                for tp in all_strikes:
                    cp = sum(max(0, tp - k) * oi for k, oi in call_oi_map.items())
                    pp = sum(max(0, k - tp) * oi for k, oi in put_oi_map.items())
                    pain[tp] = cp + pp
                max_pain_strike = min(pain, key=pain.get)  # type: ignore
            else:
                max_pain_strike = spot

            # ── Expected move (ATM straddle) ──
            atm_strike = min(chain.calls.strike.unique(), key=lambda k: abs(k - spot))
            atm_call = chain.calls[chain.calls.strike == atm_strike]
            atm_put = chain.puts[chain.puts.strike == atm_strike]
            atm_call_mid = 0.0
            atm_put_mid = 0.0
            atm_iv = 0.0
            if not atm_call.empty:
                b, a = _safe_float(atm_call.bid.iloc[0]), _safe_float(atm_call.ask.iloc[0])
                atm_call_mid = (b + a) / 2 if b + a > 0 else _safe_float(atm_call.lastPrice.iloc[0])
                atm_iv = _safe_float(atm_call.impliedVolatility.iloc[0])
            if not atm_put.empty:
                b, a = _safe_float(atm_put.bid.iloc[0]), _safe_float(atm_put.ask.iloc[0])
                atm_put_mid = (b + a) / 2 if b + a > 0 else _safe_float(atm_put.lastPrice.iloc[0])
                if atm_iv == 0:
                    atm_iv = _safe_float(atm_put.impliedVolatility.iloc[0])
            expected_move = round(atm_call_mid + atm_put_mid, 2)

            # ── IV rank (approximate from ATM IV vs HV) ──
            try:
                hist = ticker.history(period="1y")
                if not hist.empty and len(hist) > 30:
                    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                    hv_30d = float(log_ret.rolling(30).std().iloc[-1] * np.sqrt(252))
                    hv_series = log_ret.rolling(30).std().dropna() * np.sqrt(252)
                    iv_rank = int(sum(1 for v in hv_series if v <= atm_iv) / len(hv_series) * 100) if len(hv_series) > 0 else 50
                else:
                    hv_30d = 0
                    iv_rank = 50
            except Exception:
                hv_30d = 0
                iv_rank = 50

            iv_vs_hv = "premium" if atm_iv > hv_30d and hv_30d > 0 else "discount" if hv_30d > 0 else "unknown"

            # ── Unusual activity (volume > 2x OI) ──
            unusual: List[Dict] = []
            for _, r in chain.calls.iterrows():
                vol = _safe_int(r.volume)
                oi = _safe_int(r.openInterest)
                if vol > 0 and oi > 0 and vol > 2 * oi:
                    mid = (_safe_float(r.bid) + _safe_float(r.ask)) / 2 or _safe_float(r.lastPrice)
                    unusual.append({
                        "strike": float(r.strike),
                        "type": "call",
                        "volume": vol,
                        "oi": oi,
                        "ratio": round(vol / oi, 1),
                        "premium_spent": f"${int(vol * mid * 100):,}",
                    })
            for _, r in chain.puts.iterrows():
                vol = _safe_int(r.volume)
                oi = _safe_int(r.openInterest)
                if vol > 0 and oi > 0 and vol > 2 * oi:
                    mid = (_safe_float(r.bid) + _safe_float(r.ask)) / 2 or _safe_float(r.lastPrice)
                    unusual.append({
                        "strike": float(r.strike),
                        "type": "put",
                        "volume": vol,
                        "oi": oi,
                        "ratio": round(vol / oi, 1),
                        "premium_spent": f"${int(vol * mid * 100):,}",
                    })
            unusual.sort(key=lambda x: x["ratio"], reverse=True)

            # ── Top flow (biggest $ notional) ──
            top_flow: List[Dict] = []
            for _, r in chain.calls.iterrows():
                vol = _safe_int(r.volume)
                mid = (_safe_float(r.bid) + _safe_float(r.ask)) / 2 or _safe_float(r.lastPrice)
                notional = vol * mid * 100
                if notional > 100000:  # > $100K
                    top_flow.append({
                        "strike": float(r.strike),
                        "type": "call",
                        "volume": vol,
                        "notional": f"${int(notional):,}",
                        "notional_raw": notional,
                        "direction": "bullish",
                    })
            for _, r in chain.puts.iterrows():
                vol = _safe_int(r.volume)
                mid = (_safe_float(r.bid) + _safe_float(r.ask)) / 2 or _safe_float(r.lastPrice)
                notional = vol * mid * 100
                if notional > 100000:
                    top_flow.append({
                        "strike": float(r.strike),
                        "type": "put",
                        "volume": vol,
                        "notional": f"${int(notional):,}",
                        "notional_raw": notional,
                        "direction": "bearish",
                    })
            top_flow.sort(key=lambda x: x["notional_raw"], reverse=True)
            # Remove raw field before returning
            for f in top_flow:
                del f["notional_raw"]

            # ── Gamma exposure (GEX) per strike ──
            gex: List[Dict] = []
            near_strikes = [s for s in all_strikes if abs(s - spot) / spot < 0.10]
            for strike in sorted(near_strikes):
                call_oi = call_oi_map.get(strike, 0)
                put_oi = put_oi_map.get(strike, 0)
                # Call IV
                call_row = chain.calls[chain.calls.strike == strike]
                put_row = chain.puts[chain.puts.strike == strike]
                c_iv = _safe_float(call_row.impliedVolatility.iloc[0]) if not call_row.empty else atm_iv
                p_iv = _safe_float(put_row.impliedVolatility.iloc[0]) if not put_row.empty else atm_iv

                c_gamma = _bs_greeks_gamma(spot, strike, T, c_iv) if c_iv > 0 else 0
                p_gamma = _bs_greeks_gamma(spot, strike, T, p_iv) if p_iv > 0 else 0

                # GEX = (call_gamma * call_OI - put_gamma * put_OI) * 100 * spot
                gex_value = (c_gamma * call_oi - p_gamma * put_oi) * 100 * spot
                gex.append({
                    "strike": strike,
                    "gex": round(gex_value),
                    "call_oi": call_oi,
                    "put_oi": put_oi,
                })

            return json.dumps({
                "symbol": symbol,
                "spot": round(spot, 2),
                "overview": {
                    "iv_rank": iv_rank,
                    "atm_iv": round(atm_iv, 4),
                    "hv_30d": round(hv_30d, 4),
                    "iv_vs_hv": iv_vs_hv,
                    "put_call_ratio": pc_ratio,
                    "max_pain": max_pain_strike,
                    "max_pain_distance_pct": round((max_pain_strike - spot) / spot * 100, 2),
                    "expected_move": expected_move,
                    "expected_move_pct": round(expected_move / spot * 100, 2),
                    "expected_range": [round(spot - expected_move, 2), round(spot + expected_move, 2)],
                },
                "unusual_activity": unusual[:10],
                "top_flow": top_flow[:10],
                "gamma_exposure": gex,
                "nearest_expiry": {
                    "date": nearest_expiry,
                    "days": dte,
                    "atm_strike": atm_strike,
                    "total_call_oi": total_call_oi,
                    "total_put_oi": total_put_oi,
                },
                "available_expiries": friday_expiries[:8],
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": str(e)})
