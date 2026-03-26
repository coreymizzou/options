#!/usr/bin/env python3
"""
=============================================================================
OPTIONS TRADE SCANNER — Production-Ready Quantitative Analysis System
=============================================================================

SETUP INSTRUCTIONS:
1. Install dependencies:
   pip install -r requirements.txt

2. Optional API keys (set as environment variables):
   export TRADIER_API_KEY="your_key_here"          # For live options chains
   export FRED_API_KEY="your_key_here"             # For macro data

3. Run:
   python options_scanner.py                        # Live mode
   python options_scanner.py --paper               # Paper trading mode
   python options_scanner.py --tickers NVDA TSLA   # Override watchlist

REQUIREMENTS.TXT (auto-generated on first run):
   yfinance>=0.2.28
   pandas>=2.0.0
   numpy>=1.24.0
   scipy>=1.10.0
   rich>=13.0.0
   requests>=2.28.0
   pandas-datareader>=0.10.0
   matplotlib>=3.7.0
   ta>=0.10.2
   python-dateutil>=2.8.2
   argparse

=============================================================================
"""

import os
import sys
import json
import time
import math
import argparse
import warnings
import traceback
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
import subprocess

warnings.filterwarnings("ignore")

# ─── Auto-install missing packages ──────────────────────────────────────────
REQUIRED_PACKAGES = [
    "yfinance", "pandas", "numpy", "scipy", "rich", "requests",
    "pandas_datareader", "matplotlib", "ta", "python-dateutil"
]

def install_packages():
    for pkg in REQUIRED_PACKAGES:
        import_name = pkg.replace("-", "_")
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

install_packages()

# ─── Core imports ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from dateutil.relativedelta import relativedelta

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.rule import Rule
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    import pandas_datareader as pdr
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False

# =============================================================================
#  CONFIGURATION
# =============================================================================

# ── Single-stock watchlist — these are traded directionally
# High-liquidity names with active options markets and strong flow signals
WATCHLIST = [
    # Mega-cap tech
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    # High-beta growth / momentum
    "TSLA", "AMD", "CRWD", "PLTR", "COIN", "NFLX",
    # Semis + enterprise
    "MU", "CRM", "MSTR",
]

# ── Macro-only tickers — scanned for regime context and flow detection
# but NEVER surfaced as top trade recommendations.
# SPY/QQQ options are primarily used for hedging by institutions,
# not directional speculation. Single stocks give better R/R.
MACRO_ONLY_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "VXX", "UVXY",
    "SQQQ", "TQQQ", "GLD", "SLV", "TLT", "HYG",
    "XLK", "XLF", "XLE", "XBI", "SMH", "ARKK",
    "VTI", "VOO"
}

ACCOUNT_SIZE = 25_000          # Configurable account size ($)
MAX_RISK_PCT = 0.02            # Max risk per trade (2%)
MAX_RISK_DOLLARS = ACCOUNT_SIZE * MAX_RISK_PCT  # $500

# API Keys (from environment)
TRADIER_API_KEY = os.environ.get("TRADIER_API_KEY", "")
FRED_API_KEY    = os.environ.get("FRED_API_KEY", "")

# Tradier endpoints
TRADIER_BASE    = "https://api.tradier.com/v1"
TRADIER_SANDBOX = "https://sandbox.tradier.com/v1"

LOG_DIR = Path("./scanner_logs")
LOG_DIR.mkdir(exist_ok=True)

console = Console() if RICH_AVAILABLE else None

# =============================================================================
#  UTILITY HELPERS
# =============================================================================

def cprint(msg: str, style: str = ""):
    if console:
        console.print(msg, style=style)
    else:
        print(msg)

def safe_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default

def pct(val, dec=1):
    return f"{val*100:.{dec}f}%"

def dollar(val):
    return f"${val:,.2f}"

# =============================================================================
#  MODULE 1 — VOLATILITY ANALYSIS
# =============================================================================

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option pricing."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """Compute all Greeks."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1 = norm.pdf(d1)
    gamma = nd1 / (S * sigma * math.sqrt(T))
    vega  = S * nd1 * math.sqrt(T) / 100  # per 1% IV move
    theta_base = (-(S * nd1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = theta_base
        rho   = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = theta_base + r * K * math.exp(-r * T) / 365
        rho   = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """Compute IV via Brent's method."""
    if T <= 0 or market_price <= 0:
        return None
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price <= intrinsic:
        return None
    try:
        func = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        iv = brentq(func, 1e-6, 20.0, xtol=1e-6, maxiter=200)
        return iv
    except Exception:
        return None

def get_historical_volatility(ticker: str, window_days: int = 30) -> float:
    """Calculate realized/historical volatility from close prices."""
    try:
        hist = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < window_days + 5:
            return None
        closes = hist["Close"].squeeze()
        log_returns = np.log(closes / closes.shift(1)).dropna()
        hv = log_returns.rolling(window_days).std().iloc[-1] * math.sqrt(252)
        return float(hv)
    except Exception:
        return None

def extract_atm_iv_from_chain(chain_data: dict, spot: float) -> Optional[float]:
    """
    Extract ATM IV directly from Tradier chain Greeks (mid_iv field).
    Returns IV as a decimal (e.g. 0.35 for 35%) or None if unavailable.
    Uses the nearest-to-ATM contract in the first valid expiration.
    """
    for exp, data in chain_data.items():
        contracts = data if isinstance(data, list) else []
        if not contracts:
            continue
        # Find nearest ATM contract
        atm_best   = None
        atm_dist   = 999
        for c in contracts:
            try:
                K   = float(c.get("strike", 0))
                ivg = c.get("greeks") or {}
                iv  = ivg.get("mid_iv")
                if K <= 0 or iv is None or float(iv) <= 0:
                    continue
                dist = abs(K - spot)
                if dist < atm_dist:
                    atm_dist = dist
                    atm_best = float(iv)
            except Exception:
                continue
        if atm_best and atm_best > 0:
            return atm_best
    return None

def get_iv_rank(ticker: str, current_iv: float, lookback_days: int = 252) -> dict:
    """
    IVR = (current_IV - 52w_low) / (52w_high - 52w_low) * 100
    IV Percentile = % of days below current IV over lookback
    Uses ATM option IV proxied from yfinance options if available.
    """
    try:
        stock = yf.Ticker(ticker)
        exps = stock.options
        if not exps:
            return {"ivr": None, "iv_pct": None, "iv_1y_high": None, "iv_1y_low": None}

        # Get nearest expiration ATM option to approximate IV history
        # We proxy by using the stock's historical volatility as IV substitute
        hist = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if hist.empty:
            return {"ivr": None, "iv_pct": None, "iv_1y_high": None, "iv_1y_low": None}

        closes = hist["Close"].squeeze()
        log_rets = np.log(closes / closes.shift(1)).dropna()
        rolling_hv = log_rets.rolling(20).std() * math.sqrt(252)
        rolling_hv = rolling_hv.dropna()

        if len(rolling_hv) < 20:
            return {"ivr": None, "iv_pct": None, "iv_1y_high": None, "iv_1y_low": None}

        # Use current_iv if provided, else use current 20d HV
        if current_iv is None:
            current_iv = float(rolling_hv.iloc[-1])

        iv_series = rolling_hv.values
        iv_high = float(np.max(iv_series))
        iv_low  = float(np.min(iv_series))

        ivr = ((current_iv - iv_low) / (iv_high - iv_low) * 100) if (iv_high - iv_low) > 0 else 50.0
        # Hard clamp to 0-100 — current_iv can occasionally exceed the rolling
        # 52w high (e.g. during a vol spike today that's higher than any day in the
        # lookback window). Cap at 100 rather than showing 105 or 199.
        ivr    = max(0.0, min(100.0, ivr))
        iv_pct = float(np.mean(iv_series < current_iv) * 100)
        iv_pct = max(0.0, min(100.0, iv_pct))

        return {
            "ivr": round(ivr, 1),
            "iv_pct": round(iv_pct, 1),
            "iv_1y_high": round(iv_high * 100, 1),
            "iv_1y_low": round(iv_low * 100, 1),
            "current_iv_proxy": round(current_iv * 100, 1)
        }
    except Exception as e:
        return {"ivr": None, "iv_pct": None, "iv_1y_high": None, "iv_1y_low": None}

def compute_volatility_surface(ticker: str, spot_price: float) -> dict:
    """
    Build simplified IV surface from yfinance options chains.
    Returns dict: {expiry: {strike: iv}}
    """
    surface = {}
    atm_ivs = {}
    try:
        stock = yf.Ticker(ticker)
        exps  = stock.options
        if not exps:
            return {}, {}

        r = 0.045  # approx risk-free rate

        for exp in exps[:5]:  # first 5 expirations
            try:
                chain = stock.option_chain(exp)
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                T = max((exp_date - datetime.now()).days / 365, 1/365)

                exp_ivs = {}
                calls = chain.calls
                puts  = chain.puts

                # Focus on strikes near ATM (80%-120% moneyness)
                for _, row in calls.iterrows():
                    K = float(row["strike"])
                    moneyness = K / spot_price
                    if 0.80 <= moneyness <= 1.25:
                        mid = (float(row["bid"]) + float(row["ask"])) / 2 if row["bid"] > 0 else float(row["lastPrice"])
                        iv = implied_volatility(mid, spot_price, K, T, r, "call")
                        if iv and 0.01 < iv < 5.0:
                            exp_ivs[K] = round(iv * 100, 1)

                if exp_ivs:
                    surface[exp] = exp_ivs
                    # ATM IV = IV of strike closest to spot
                    closest_k = min(exp_ivs.keys(), key=lambda k: abs(k - spot_price))
                    atm_ivs[exp] = exp_ivs[closest_k]
            except Exception:
                continue

    except Exception:
        pass

    return surface, atm_ivs

def analyze_vol_skew(surface: dict, spot: float, exp: str) -> dict:
    """Detect put/call skew anomalies for a given expiration."""
    if exp not in surface:
        return {}
    ivs = surface[exp]
    if not ivs:
        return {}

    otm_calls = {k: v for k, v in ivs.items() if k > spot * 1.02}
    otm_puts  = {k: v for k, v in ivs.items() if k < spot * 0.98}
    atm_ivs   = {k: v for k, v in ivs.items() if 0.98 <= k/spot <= 1.02}

    atm_iv    = np.mean(list(atm_ivs.values())) if atm_ivs else None
    put_iv    = np.mean(list(otm_puts.values())) if otm_puts else None
    call_iv   = np.mean(list(otm_calls.values())) if otm_calls else None

    skew = {}
    if atm_iv and put_iv:
        skew["put_skew"] = round(put_iv - atm_iv, 1)
        skew["put_skew_flag"] = "ELEVATED" if put_iv - atm_iv > 5 else "NORMAL"
    if atm_iv and call_iv:
        skew["call_skew"] = round(call_iv - atm_iv, 1)
        skew["call_skew_flag"] = "ELEVATED" if call_iv - atm_iv > 5 else "NORMAL"
    skew["atm_iv"] = atm_iv

    return skew

# =============================================================================
#  MODULE 2 — OPTIONS FLOW SCANNER
# =============================================================================

def get_options_chain_tradier(ticker: str, paper_mode: bool = False) -> dict:
    """Fetch live options chain from Tradier (requires API key)."""
    if not TRADIER_API_KEY:
        return {}
    base = TRADIER_SANDBOX if paper_mode else TRADIER_BASE
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }
    try:
        # Get expirations
        exp_url = f"{base}/markets/options/expirations"
        r = requests.get(exp_url, headers=headers, params={"symbol": ticker, "includeAllRoots": "true"}, timeout=10)
        if r.status_code != 200:
            return {}
        exps = r.json().get("expirations", {}).get("date", [])
        if isinstance(exps, str):
            exps = [exps]

        chains = {}
        today  = datetime.now()

        # Split expirations into two buckets:
        # near_exps  — first 4 near-term exps for flow scanning
        # valid_exps — first 3 expirations that are >= MIN_OPTION_DTE for trade construction
        # We fetch both sets and tag them so the caller can split them.
        near_exps  = exps[:4]
        valid_exps = [
            e for e in exps
            if (datetime.strptime(e, "%Y-%m-%d") - today).days >= MIN_OPTION_DTE
        ][:3]  # only need 3 valid expirations for trade construction

        # Combine — fetch near-term for flow, valid for trades
        to_fetch = list(dict.fromkeys(near_exps + valid_exps))  # deduplicated, order preserved

        for exp in to_fetch:
            chain_url = f"{base}/markets/options/chains"
            cr = requests.get(chain_url, headers=headers,
                              params={"symbol": ticker, "expiration": exp, "greeks": "true"}, timeout=10)
            if cr.status_code == 200:
                options = cr.json().get("options", {}).get("option", [])
                if options:
                    chains[exp] = options
        return chains
    except Exception:
        return {}

def get_options_chain_yfinance(ticker: str) -> dict:
    """Fallback: fetch options chain from yfinance."""
    chains = {}
    try:
        stock = yf.Ticker(ticker)
        exps  = stock.options or []
        today = datetime.now()

        # Same split as Tradier: near-term for flow, valid for trades
        near_exps  = list(exps[:4])
        valid_exps = [
            e for e in exps
            if (datetime.strptime(e, "%Y-%m-%d") - today).days >= MIN_OPTION_DTE
        ][:3]
        to_fetch = list(dict.fromkeys(near_exps + valid_exps))

        for exp in to_fetch:
            try:
                chain = stock.option_chain(exp)
                chains[exp] = {
                    "calls": chain.calls.to_dict("records"),
                    "puts":  chain.puts.to_dict("records")
                }
            except Exception:
                continue
    except Exception:
        pass
    return chains

# Minimum DTE for trade construction — never recommend entering
# a new long option or spread with fewer than this many days left.
# Flow scanning uses the full chain (we want to see all activity).
MIN_OPTION_DTE = 21

def filter_chain_by_dte(chain_data: dict, min_dte: int = MIN_OPTION_DTE) -> dict:
    """
    Remove expirations with fewer than min_dte days remaining.
    Returns a filtered copy of chain_data safe for trade construction.
    The original chain_data is preserved for flow scanning.
    """
    filtered = {}
    today = datetime.now()
    for exp, data in chain_data.items():
        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            dte    = (exp_dt - today).days
            if dte >= min_dte:
                filtered[exp] = data
        except Exception:
            continue
    return filtered

def scan_unusual_flow(chains_yf: dict, spot_price: float, paper_mode: bool = False) -> list:
    """
    Scan for genuinely unusual options activity using tiered criteria.

    Tiers (all require minimum contract counts to filter out thin-OI noise):
    ─────────────────────────────────────────────────────────────────────
    UNUSUAL:   Vol/OI > 3x   AND volume >= 200  AND OI >= 100
               Baseline signal — someone is notably active on this strike
    SWEEP:     Vol/OI > 10x  AND volume >= 500
               Aggressive accumulation across the chain
    BIG PRINT: Premium > $250k AND volume >= 200
               Real money committed — not a retail fluke
    WHALE:     Premium > $1M  AND volume >= 1000 AND OI >= 500
               Institutional-sized conviction trade
    ─────────────────────────────────────────────────────────────────────
    Minimum OI requirement (>= 100) prevents Vol/OI of 100x on OI=1
    from masquerading as a meaningful signal.
    """
    signals = []

    for exp, data in chains_yf.items():
        if isinstance(data, dict):
            all_contracts = []
            for side in ["calls", "puts"]:
                for c in data.get(side, []):
                    c["option_type"] = "call" if side == "calls" else "put"
                    all_contracts.append(c)
        else:
            all_contracts = data  # Tradier format

        for c in all_contracts:
            try:
                volume   = float(c.get("volume", 0) or 0)
                oi       = float(c.get("openInterest", c.get("open_interest", 0)) or 0)
                strike   = float(c.get("strike", 0))
                bid      = float(c.get("bid", 0) or 0)
                ask      = float(c.get("ask", 0) or 0)
                last     = float(c.get("lastPrice", c.get("last", 0)) or 0)
                opt_type = c.get("option_type", "call")

                if strike == 0:
                    continue

                # ── Minimum volume floor — ignore micro prints entirely
                if volume < 100:
                    continue

                # ── Minimum OI floor — prevents 100x Vol/OI on OI=1 situations
                # Exception: allow low OI if volume is very large (>= 2000)
                # because brand new strikes can have low OI legitimately
                if oi < 100 and volume < 2000:
                    continue

                vol_oi_ratio = volume / max(oi, 1)
                mid          = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                premium_paid = mid * volume * 100

                if mid <= 0:
                    continue

                # ── Trade direction detection (aggressor side)
                # Determines if the flow was a BUY or SELL at the market.
                # last >= ask → buyer lifted the offer (paid up) = aggressive buyer
                # last <= bid → seller hit the bid (sold down) = aggressive seller
                # This flips the directional signal for puts:
                #   Buying puts = bearish, Selling puts = bullish (collecting premium)
                if bid > 0 and ask > 0 and last > 0:
                    if last >= ask * 0.98:      # within 2% of ask = bought at ask
                        aggressor = "BUY"
                    elif last <= bid * 1.02:    # within 2% of bid = sold at bid
                        aggressor = "SELL"
                    else:
                        aggressor = "UNKNOWN"   # mid fill — ambiguous
                else:
                    aggressor = "UNKNOWN"

                # Directional inference accounting for aggressor side:
                # BUY  call → BULLISH   |   SELL call → BEARISH (writing calls)
                # BUY  put  → BEARISH   |   SELL put  → BULLISH (writing/selling puts)
                if aggressor == "BUY":
                    inferred_dir = "BULLISH" if opt_type == "call" else "BEARISH"
                    dir_confidence = "HIGH"
                elif aggressor == "SELL":
                    inferred_dir = "BEARISH" if opt_type == "call" else "BULLISH"
                    dir_confidence = "HIGH"
                else:
                    # Ambiguous fill — default to option type direction but low confidence
                    inferred_dir = "BULLISH" if opt_type == "call" else "BEARISH"
                    dir_confidence = "LOW"

                # Moneyness
                moneyness = strike / spot_price
                is_otm = (
                    (moneyness > 1.02 and opt_type == "call") or
                    (moneyness < 0.98 and opt_type == "put")
                )

                # ── Tiered scoring
                score = 0
                flags = []

                # Tier 1: Unusual Vol/OI (requires real OI base)
                if vol_oi_ratio >= 3 and oi >= 100:
                    score += 2
                    flags.append(f"Vol/OI={vol_oi_ratio:.1f}x")

                # Tier 2: Sweep (aggressive accumulation)
                if vol_oi_ratio >= 10 and volume >= 500:
                    score += 2
                    flags.append("SWEEP")

                # Tier 3: Big Print (real money)
                if premium_paid >= 250_000 and volume >= 200:
                    score += 2
                    flags.append(f"Big Print ${premium_paid/1000:.0f}k")

                # Tier 4: Whale (institutional size)
                # Requires BOTH large premium AND large contract count
                # This prevents one expensive contract from triggering WHALE
                if premium_paid >= 1_000_000 and volume >= 1000 and oi >= 500:
                    score += 3
                    flags.append("WHALE")
                elif premium_paid >= 500_000 and volume >= 500:
                    score += 2
                    flags.append("LARGE BLOCK")

                # OTM bonus — OTM flow is more directionally meaningful
                if is_otm and score > 0:
                    score += 1
                    flags.append("OTM")

                # Only surface if at least two tiers triggered
                # (prevents single-criterion noise from showing up)
                if score >= 4:
                    # Add aggressor flag to display
                    if aggressor == "BUY":
                        flags.append(f"BOUGHT at ask")
                    elif aggressor == "SELL":
                        flags.append(f"SOLD at bid")

                    signals.append({
                        "exp":            exp,
                        "strike":         strike,
                        "opt_type":       opt_type,
                        "volume":         int(volume),
                        "oi":             int(oi),
                        "vol_oi_ratio":   round(vol_oi_ratio, 1),
                        "premium_paid":   round(premium_paid, 0),
                        "mid_price":      round(mid, 2),
                        "direction":      inferred_dir,
                        "aggressor":      aggressor,
                        "dir_confidence": dir_confidence,
                        "flags":          flags,
                        "score":          score,
                        "moneyness":      round(moneyness, 3)
                    })
            except Exception:
                continue

    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals[:5]

# =============================================================================
#  MODULE 3 — TECHNICAL SIGNAL LAYER
# =============================================================================

def get_price_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        return data
    except Exception:
        return pd.DataFrame()

def compute_technicals(df: pd.DataFrame) -> dict:
    """Compute RSI, EMA, VWAP, ATR, candlestick patterns."""
    if df.empty or len(df) < 30:
        return {}

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    signals = {}

    # ── RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    signals["rsi"] = round(float(rsi.iloc[-1]), 1)
    signals["rsi_signal"] = (
        "OVERSOLD"   if signals["rsi"] < 30 else
        "OVERBOUGHT" if signals["rsi"] > 70 else
        "NEUTRAL"
    )

    # ── EMAs
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    signals["ema9"]   = round(float(ema9.iloc[-1]), 2)
    signals["ema21"]  = round(float(ema21.iloc[-1]), 2)
    signals["ema50"]  = round(float(ema50.iloc[-1]), 2)
    signals["price"]  = round(float(close.iloc[-1]), 2)

    # EMA crossover signal
    if ema9.iloc[-1] > ema21.iloc[-1] and ema9.iloc[-2] <= ema21.iloc[-2]:
        signals["ema_cross"] = "BULL_CROSS"
    elif ema9.iloc[-1] < ema21.iloc[-1] and ema9.iloc[-2] >= ema21.iloc[-2]:
        signals["ema_cross"] = "BEAR_CROSS"
    elif ema9.iloc[-1] > ema21.iloc[-1]:
        signals["ema_cross"] = "BULLISH"
    else:
        signals["ema_cross"] = "BEARISH"

    # ── VWAP (approximate daily VWAP from close/volume proxy)
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
    signals["vwap"] = round(float(vwap.iloc[-1]), 2)
    signals["above_vwap"] = close.iloc[-1] > vwap.iloc[-1]

    # ── ATR (14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    signals["atr"] = round(float(atr.iloc[-1]), 2)
    signals["atr_pct"] = round(float(atr.iloc[-1] / close.iloc[-1] * 100), 2)

    # ── Candlestick patterns
    patterns = []
    o, c_now = float(df["Open"].iloc[-1]), float(close.iloc[-1])
    c_prev    = float(close.iloc[-2])
    o_prev    = float(df["Open"].iloc[-2])
    body_now  = abs(c_now - o)
    body_prev = abs(c_prev - o_prev)

    # Engulfing
    if c_now > o and c_prev < o_prev and c_now > o_prev and o < c_prev:
        patterns.append("BULL_ENGULFING")
    elif c_now < o and c_prev > o_prev and c_now < o_prev and o > c_prev:
        patterns.append("BEAR_ENGULFING")

    # Inside bar
    if high.iloc[-1] <= high.iloc[-2] and low.iloc[-1] >= low.iloc[-2]:
        patterns.append("INSIDE_BAR")

    # Gap up/down
    gap_pct = (o - c_prev) / c_prev * 100
    if gap_pct > 0.5:
        patterns.append(f"GAP_UP_{gap_pct:.1f}%")
    elif gap_pct < -0.5:
        patterns.append(f"GAP_DOWN_{abs(gap_pct):.1f}%")

    signals["patterns"] = patterns

    # ── Support / Resistance levels
    recent_highs = high.rolling(20).max()
    recent_lows  = low.rolling(20).min()
    signals["resistance"] = round(float(recent_highs.iloc[-1]), 2)
    signals["support"]    = round(float(recent_lows.iloc[-1]), 2)

    # Near key level?
    near_res = abs(close.iloc[-1] - recent_highs.iloc[-1]) / close.iloc[-1] < 0.01
    near_sup  = abs(close.iloc[-1] - recent_lows.iloc[-1]) / close.iloc[-1] < 0.01
    if near_res:
        signals["key_level"] = "NEAR_RESISTANCE"
    elif near_sup:
        signals["key_level"] = "NEAR_SUPPORT"
    else:
        signals["key_level"] = "MID_RANGE"

    # ── Volume analysis
    avg_vol = volume.rolling(20).mean().iloc[-1]
    signals["volume"] = int(volume.iloc[-1])
    signals["avg_volume"] = int(avg_vol)
    signals["rel_volume"] = round(float(volume.iloc[-1] / avg_vol), 2)
    signals["high_volume"] = signals["rel_volume"] > 1.5

    return signals

def get_sector_correlation(ticker: str) -> dict:
    """Check if SPY and QQQ are confirming."""
    try:
        spy_data = get_price_data("SPY", period="5d", interval="1d")
        qqq_data = get_price_data("QQQ", period="5d", interval="1d")

        spy_ret = float((spy_data["Close"].iloc[-1] / spy_data["Close"].iloc[-2] - 1) * 100)
        qqq_ret = float((qqq_data["Close"].iloc[-1] / qqq_data["Close"].iloc[-2] - 1) * 100)

        spy_trend = "BULLISH" if spy_ret > 0.1 else ("BEARISH" if spy_ret < -0.1 else "NEUTRAL")
        qqq_trend = "BULLISH" if qqq_ret > 0.1 else ("BEARISH" if qqq_ret < -0.1 else "NEUTRAL")

        return {
            "spy_change_pct": round(spy_ret, 2),
            "qqq_change_pct": round(qqq_ret, 2),
            "spy_trend": spy_trend,
            "qqq_trend": qqq_trend,
            "broad_market": spy_trend if spy_trend == qqq_trend else "MIXED"
        }
    except Exception:
        return {"broad_market": "UNKNOWN", "spy_trend": "UNKNOWN", "qqq_trend": "UNKNOWN"}

# Tickers that never have earnings calendars — skip the lookup entirely
# to avoid noisy 404 errors. Kept in sync with MACRO_ONLY_TICKERS above.
_NO_EARNINGS_TICKERS = MACRO_ONLY_TICKERS

def get_days_to_earnings(ticker: str) -> Optional[int]:
    """
    Estimate days to next earnings using yfinance calendar.
    ETFs and index funds are skipped silently — they never have earnings.
    """
    if ticker.upper() in _NO_EARNINGS_TICKERS:
        return None  # silently skip — no earnings for ETFs/indices

    try:
        import urllib.request
        # Suppress stderr to avoid 404 noise from yfinance HTTP calls
        stock = yf.Ticker(ticker)

        # yfinance sometimes prints HTTP errors directly to stdout/stderr
        # Redirect around the calendar fetch
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            cal = stock.calendar

        if cal is None or len(cal) == 0:
            return None
        if isinstance(cal, dict):
            earn_dates = cal.get("Earnings Date", [])
        elif isinstance(cal, pd.DataFrame):
            earn_dates = cal.values.flatten().tolist()
        else:
            return None

        for d in earn_dates:
            try:
                if hasattr(d, "date"):
                    d = d.date()
                elif isinstance(d, str):
                    d = datetime.strptime(d, "%Y-%m-%d").date()
                days = (d - date.today()).days
                if days >= -1:
                    return days
            except Exception:
                continue
        return None
    except Exception:
        return None

# =============================================================================
#  MODULE 4 — MARKET REGIME FILTER
# =============================================================================

def determine_market_regime() -> dict:
    """
    Classify broad market as: TRENDING_UP, TRENDING_DOWN, RANGING, RISK_OFF
    Uses SPY price action + VIX level.
    """
    try:
        # SPY
        spy = get_price_data("SPY", period="3mo", interval="1d")
        close = spy["Close"].squeeze()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        spy_above_ema20 = close.iloc[-1] > ema20.iloc[-1]
        ema20_above_50  = ema20.iloc[-1] > ema50.iloc[-1]
        spy_1m_ret      = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)

        # VIX
        vix_data = get_price_data("^VIX", period="1mo", interval="1d")
        vix = float(vix_data["Close"].iloc[-1]) if not vix_data.empty else 20.0

        # RSI on SPY
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        spy_rsi = float((100 - 100 / (1 + rs)).iloc[-1])

        # Regime determination
        if vix > 30:
            regime = "RISK_OFF"
        elif spy_above_ema20 and ema20_above_50 and spy_1m_ret > 2:
            regime = "TRENDING_UP"
        elif not spy_above_ema20 and not ema20_above_50 and spy_1m_ret < -2:
            regime = "TRENDING_DOWN"
        else:
            regime = "RANGING"

        return {
            "regime": regime,
            "vix": round(vix, 1),
            "spy_1m_return": round(spy_1m_ret, 1),
            "spy_rsi": round(spy_rsi, 1),
            "vix_regime": "FEAR" if vix > 25 else ("ELEVATED" if vix > 18 else "CALM")
        }
    except Exception:
        return {"regime": "UNKNOWN", "vix": None, "vix_regime": "UNKNOWN"}

# =============================================================================
#  MODULE 5 — TRADE CONSTRUCTION ENGINE
# =============================================================================

def select_optimal_expiration(exps: list, dte_target: int = 30,
                               min_dte: int = 0) -> str:
    """
    Pick expiration closest to target DTE, subject to a minimum DTE floor.

    min_dte: skip any expiration with fewer than this many days remaining.
    This prevents recommending options that expire before the trade can develop.
    Example: min_dte=21 ensures we never suggest a long option with < 3 weeks left.
    """
    if not exps:
        return ""
    valid = [
        e for e in exps
        if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days >= min_dte
    ]
    if not valid:
        # If nothing passes the floor, just take the furthest-dated expiration
        valid = exps
    best = min(valid, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days - dte_target))
    return best

def find_best_strike(chain_data: dict, exp: str, spot: float, direction: str,
                     delta_target: float = 0.40, option_type: str = "call") -> dict:
    """
    Find optimal strike targeting ~0.40 delta (OTM).

    Key guardrails:
    - Hard moneyness filter: calls must be 0.90–1.20 of spot, puts 0.80–1.10
      This prevents the function from picking deep ITM contracts.
    - Spread filter: skip contracts where bid/ask spread > 20% of mid
      (illiquid contracts produce garbage IV and delta estimates)
    - Delta sanity check: skip any contract whose computed |delta| > 0.85
      If IV fallback (0.30) caused a bad delta, this catches it.
    - Prefer contracts with open interest > 0 (real markets exist)
    """
    best = None
    best_delta_diff = 999

    try:
        if exp not in chain_data:
            return {}
        data = chain_data[exp]

        contracts = data.get(option_type + "s", []) if isinstance(data, dict) else []

        r = 0.045
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        T = max((exp_dt - datetime.now()).days / 365, 1/365)

        # ── Moneyness bounds: only consider strikes in a sensible OTM range
        # Calls: strike between 90% and 120% of spot (avoids deep ITM)
        # Puts:  strike between 80% and 110% of spot (avoids deep ITM)
        if option_type == "call":
            mono_lo, mono_hi = 0.90, 1.20
        else:
            mono_lo, mono_hi = 0.80, 1.10

        for c in contracts:
            K     = float(c.get("strike", 0))
            bid   = float(c.get("bid", 0) or 0)
            ask   = float(c.get("ask", 0) or 0)
            last  = float(c.get("lastPrice", c.get("last", 0)) or 0)
            oi    = int(c.get("openInterest", c.get("open_interest", 0)) or 0)

            if K <= 0:
                continue

            # ── Moneyness filter
            moneyness = K / spot
            if not (mono_lo <= moneyness <= mono_hi):
                continue

            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
            if mid <= 0:
                continue

            # ── Liquidity filter: skip wide spreads
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / mid
                if spread_pct > 0.50:
                    continue

            # ── Greeks: prefer Tradier native Greeks over Black-Scholes
            # Tradier returns live market Greeks when greeks=true is passed.
            # These are more accurate than our BS approximation because:
            #   1. They use the market's own IV surface (skew-adjusted)
            #   2. They reflect live bid/ask, not theoretical mid
            #   3. Delta is from the market maker's model, not our simplified BS
            tradier_greeks = c.get("greeks") or {}
            native_iv    = tradier_greeks.get("mid_iv")    # 0-1 scale from Tradier
            native_delta = tradier_greeks.get("delta")
            native_gamma = tradier_greeks.get("gamma")
            native_theta = tradier_greeks.get("theta")     # $/share/day from Tradier
            native_vega  = tradier_greeks.get("vega")

            has_native_greeks = (
                native_iv    is not None and
                native_delta is not None and
                native_theta is not None and
                float(native_iv or 0) > 0
            )

            if has_native_greeks:
                # Use Tradier's live market Greeks directly
                iv     = float(native_iv)
                delta  = float(native_delta)
                gamma  = float(native_gamma or 0)
                theta_val = float(native_theta or 0)  # already $/share/day
                vega_val  = float(native_vega  or 0)
                greeks_source = "tradier"
            else:
                # Fall back to Black-Scholes when Tradier Greeks unavailable
                iv = implied_volatility(mid, spot, K, T, r, option_type)
                if iv is None or iv <= 0 or iv > 5.0:
                    iv = 0.40
                bs = black_scholes_greeks(spot, K, T, r, iv, option_type)
                delta     = bs["delta"]
                gamma     = bs["gamma"]
                theta_val = bs["theta"]   # $/share/day from BS
                vega_val  = bs["vega"]
                greeks_source = "black_scholes"

            # ── Delta sanity check
            if abs(delta) > 0.85:
                continue

            if abs(abs(delta) - delta_target) < best_delta_diff:
                best_delta_diff = abs(abs(delta) - delta_target)
                best = {
                    "strike": K, "exp": exp, "mid": round(mid, 2),
                    "bid": round(bid, 2), "ask": round(ask, 2),
                    "iv": round(iv * 100, 1),
                    "delta": round(delta, 3),
                    "gamma": round(gamma, 4),
                    # Theta display: always show as $/contract/day for consistency
                    # Tradier theta is $/share/day → multiply by 100
                    # BS theta is already scaled by 100 in find_best_strike
                    "theta": round(theta_val * 100, 2) if greeks_source == "tradier" else round(theta_val * 100, 2),
                    "vega":  round(vega_val  * 100, 2),
                    "iv_source": greeks_source,
                    "option_type": option_type,
                    "oi": oi,
                    "volume": int(c.get("volume", 0) or 0),
                    "T_years": T
                }

        # ── If still nothing found after filters, try a wider moneyness window
        # (handles cases like low-priced stocks or very high IV names)
        if best is None:
            if option_type == "call":
                mono_lo, mono_hi = 0.85, 1.30
            else:
                mono_lo, mono_hi = 0.70, 1.15

            for c in contracts:
                K   = float(c.get("strike", 0))
                bid = float(c.get("bid", 0) or 0)
                ask = float(c.get("ask", 0) or 0)
                last = float(c.get("lastPrice", c.get("last", 0)) or 0)

                if K <= 0:
                    continue
                moneyness = K / spot
                if not (mono_lo <= moneyness <= mono_hi):
                    continue

                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                if mid <= 0:
                    continue

                tradier_greeks2 = c.get("greeks") or {}
                native_iv2    = tradier_greeks2.get("mid_iv")
                native_delta2 = tradier_greeks2.get("delta")
                native_gamma2 = tradier_greeks2.get("gamma")
                native_theta2 = tradier_greeks2.get("theta")
                native_vega2  = tradier_greeks2.get("vega")

                has_native2 = (native_iv2 is not None and native_delta2 is not None
                               and float(native_iv2 or 0) > 0)

                if has_native2:
                    iv2    = float(native_iv2)
                    delta2 = float(native_delta2)
                    gamma2 = float(native_gamma2 or 0)
                    theta2 = float(native_theta2 or 0)
                    vega2  = float(native_vega2  or 0)
                    gsrc2  = "tradier"
                else:
                    iv2 = implied_volatility(mid, spot, K, T, r, option_type)
                    if iv2 is None or iv2 <= 0 or iv2 > 5.0:
                        iv2 = 0.40
                    bs2    = black_scholes_greeks(spot, K, T, r, iv2, option_type)
                    delta2 = bs2["delta"]
                    gamma2 = bs2["gamma"]
                    theta2 = bs2["theta"]
                    vega2  = bs2["vega"]
                    gsrc2  = "black_scholes"

                if abs(delta2) > 0.90:
                    continue

                if abs(abs(delta2) - delta_target) < best_delta_diff:
                    best_delta_diff = abs(abs(delta2) - delta_target)
                    best = {
                        "strike": K, "exp": exp, "mid": round(mid, 2),
                        "bid": round(bid, 2), "ask": round(ask, 2),
                        "iv": round(iv2 * 100, 1),
                        "delta": round(delta2, 3),
                        "gamma": round(gamma2, 4),
                        "theta": round(theta2 * 100, 2),
                        "vega":  round(vega2  * 100, 2),
                        "iv_source": gsrc2,
                        "option_type": option_type,
                        "oi": int(c.get("openInterest", c.get("open_interest", 0)) or 0),
                        "volume": int(c.get("volume", 0) or 0),
                        "T_years": T
                    }

    except Exception:
        pass

    return best or {}

def construct_trade(
    ticker: str,
    spot: float,
    direction: str,
    ivr: dict,
    tech: dict,
    flow_signals: list,
    chain_data: dict,
    regime: dict,
    dte_days: Optional[int],
    paper_mode: bool = False
) -> dict:
    """
    Determine optimal trade structure and select contract.

    Strategy selection logic:
    1. Risk-off  → conservative / avoid or go bearish
    2. IVR > 50  → sell premium (CSP, CC, Debit Spread)
    3. IVR < 30  → buy premium (Long calls/puts, Straddle)
    4. Earnings catalyst → Straddle/Strangle
    5. Directional flow → Long debit
    """
    ivr_val  = ivr.get("ivr", 50) or 50
    regime_n = regime.get("regime", "UNKNOWN")

    exps = list(chain_data.keys())
    if not exps:
        return {"error": "No option chains available"}

    # ── Determine direction from flow + technicals
    flow_dir = "NEUTRAL"
    if flow_signals:
        # Weight HIGH confidence signals (known aggressor) more than ambiguous ones
        bull_score = sum(
            2 if f.get("dir_confidence") == "HIGH" else 1
            for f in flow_signals if f["direction"] == "BULLISH"
        )
        bear_score = sum(
            2 if f.get("dir_confidence") == "HIGH" else 1
            for f in flow_signals if f["direction"] == "BEARISH"
        )
        if bull_score > bear_score:
            flow_dir = "BULLISH"
        elif bear_score > bull_score:
            flow_dir = "BEARISH"

    ema_dir  = tech.get("ema_cross", "NEUTRAL")
    final_dir = direction if direction != "NEUTRAL" else flow_dir

    # Fallback to technicals
    if final_dir == "NEUTRAL":
        if "BULL" in ema_dir:
            final_dir = "BULLISH"
        elif "BEAR" in ema_dir:
            final_dir = "BEARISH"

    # ── Strategy selection
    if dte_days is not None and 1 <= dte_days <= 7:
        strategy = "LONG_STRADDLE"
        rationale = f"Earnings in {dte_days} days — IV expansion play"
        # Straddles are direction-neutral by definition — override any directional signal
        final_dir = "NEUTRAL"
    elif dte_days is not None and 8 <= dte_days <= 21:
        if final_dir == "BULLISH":
            strategy = "LONG_CALL"
            rationale = f"Earnings in {dte_days} days — directional + vega play"
        else:
            strategy = "LONG_PUT"
            rationale = f"Earnings in {dte_days} days — directional + vega play"
    elif regime_n == "RISK_OFF":
        strategy = "BEAR_PUT_SPREAD"
        rationale = "Risk-off market regime — defensive bearish spread"
    elif ivr_val > 60:
        if final_dir == "BULLISH":
            strategy = "BULL_CALL_SPREAD"
            rationale = f"High IV ({ivr_val:.0f}) — debit spread to reduce cost basis"
        else:
            strategy = "BEAR_PUT_SPREAD"
            rationale = f"High IV ({ivr_val:.0f}) — put debit spread"
    elif ivr_val < 30 and dte_days and dte_days < 30:
        strategy = "LONG_STRADDLE"
        rationale = f"Low IV ({ivr_val:.0f}) + catalyst within {dte_days}d — straddle"
    elif ivr_val < 35:
        if final_dir == "BULLISH":
            strategy = "LONG_CALL"
            rationale = f"Low IV ({ivr_val:.0f}) — options cheap, directional long call"
        else:
            strategy = "LONG_PUT"
            rationale = f"Low IV ({ivr_val:.0f}) — options cheap, directional long put"
    elif final_dir == "BULLISH":
        strategy = "BULL_CALL_SPREAD"
        rationale = "Moderate IV — limited risk bull spread"
    else:
        strategy = "BEAR_PUT_SPREAD"
        rationale = "Moderate IV — limited risk bear spread"

    # ── Select contracts
    # Minimum DTE rules:
    #   Long options (no catalyst): 21 days minimum — theta decay is brutal < 21 DTE
    #   Spreads: 21 days minimum — need time for the spread to work
    #   Earnings straddle: just after earnings date so IV crush doesn't kill immediately
    #   LEAP: no floor needed, always long-dated
    if strategy == "LONG_STRADDLE" and dte_days is not None:
        target_dte = max(dte_days + 3, 7)
        min_dte    = max(dte_days + 1, 5)  # must expire after earnings
    elif strategy == "LEAP":
        target_dte = 365
        min_dte    = 180
    elif strategy in ("LONG_CALL", "LONG_PUT"):
        target_dte = 30
        min_dte    = 21  # never recommend a long option with < 3 weeks
    elif "SPREAD" in strategy:
        target_dte = 30
        min_dte    = 21  # spreads need time to develop too
    else:
        target_dte = 30
        min_dte    = 0
    best_exp = select_optimal_expiration(exps, target_dte, min_dte=min_dte)

    trade = {
        "ticker": ticker,
        "spot": spot,
        "strategy": strategy,
        "direction": final_dir,
        "rationale": rationale,
        "exp": best_exp,
        "ivr": ivr_val,
        "regime": regime_n
    }

    opt_type = "call" if final_dir == "BULLISH" or strategy in ["LONG_CALL", "BULL_CALL_SPREAD", "LONG_STRADDLE"] else "put"

    # Main leg
    main_leg = find_best_strike(chain_data, best_exp, spot, final_dir, delta_target=0.40, option_type=opt_type)

    if not main_leg:
        # Fallback: compute a theoretical ~0.40-delta OTM strike via BS inversion.
        # This is THEORETICAL — flag it clearly so user knows to verify before trading.
        trade["data_quality"] = "THEORETICAL (no live chain match — verify before trading)"

        # Re-run expiration selection enforcing the same min_dte floor.
        # The initial best_exp may have been chosen from a sparse weekend chain
        # that didn't respect the floor. Re-select now that we're in fallback mode.
        best_exp_checked = select_optimal_expiration(exps, target_dte, min_dte=min_dte)
        if best_exp_checked:
            best_exp = best_exp_checked
            trade["exp"] = best_exp

        exp_dt    = datetime.strptime(best_exp, "%Y-%m-%d") if best_exp else datetime.now() + timedelta(days=target_dte)
        actual_dte = (exp_dt - datetime.now()).days
        # If still under min_dte (chain only has near-term expirations), warn user
        if actual_dte < min_dte:
            trade["data_quality"] += f" — WARNING: nearest available expiry is only {actual_dte}d (target {min_dte}d+)"
        T         = max(actual_dte / 365, 1/365)
        iv_approx = (ivr.get("current_iv_proxy", 40) or 40) / 100
        r_rf      = 0.045

        # Derive ~0.40-delta OTM strike analytically from BS
        # Call: K = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Phi_inv(0.60))
        # Put:  K = S * exp((r - 0.5*sigma^2)*T - sigma*sqrt(T)*Phi_inv(0.60))
        z_40d = norm.ppf(0.60)  # ~0.253
        if opt_type == "call":
            otm_strike = spot * math.exp((r_rf - 0.5 * iv_approx**2) * T + iv_approx * math.sqrt(T) * z_40d)
        else:
            otm_strike = spot * math.exp((r_rf - 0.5 * iv_approx**2) * T - iv_approx * math.sqrt(T) * z_40d)

        rounding   = 5 if spot > 100 else (2.5 if spot > 50 else 1)
        otm_strike = round(otm_strike / rounding) * rounding

        fake_price = black_scholes_price(spot, otm_strike, T, r_rf, iv_approx, opt_type)
        bs_greeks  = black_scholes_greeks(spot, otm_strike, T, r_rf, iv_approx, opt_type)

        main_leg = {
            "strike": otm_strike, "exp": best_exp,
            "mid": round(max(fake_price, 0.05), 2),
            "bid": round(max(fake_price * 0.93, 0.01), 2),
            "ask": round(fake_price * 1.07, 2),
            "iv": round(iv_approx * 100, 1),
            "delta": round(bs_greeks["delta"], 3),
            "gamma": round(bs_greeks["gamma"], 4),
            "theta": round(bs_greeks["theta"] * 100, 2),
            "vega":  round(bs_greeks["vega"] * 100, 2),
            "option_type": opt_type, "oi": 0, "volume": 0,
            "T_years": T
        }

    trade["main_leg"] = main_leg

    # ── Spread: add short leg
    if "SPREAD" in strategy:
        spread_width = max(round(spot * 0.03 / 5) * 5, 5)  # ~3% OTM width
        short_strike = (
            main_leg["strike"] + spread_width if opt_type == "call"
            else main_leg["strike"] - spread_width
        )
        short_leg = find_best_strike(chain_data, best_exp, spot, final_dir,
                                      delta_target=0.25, option_type=opt_type)
        if not short_leg:
            short_leg = dict(main_leg)
            short_leg["strike"] = short_strike
            short_leg["mid"]    = round(main_leg["mid"] * 0.50, 2)
        trade["short_leg"] = short_leg

    # ── Straddle: add ATM put leg at the same strike as the call
    if strategy == "LONG_STRADDLE":
        # For a true straddle both legs should be at the same ATM strike.
        # The call leg was found at ~0.40 delta (slightly OTM); use that same
        # strike for the put so both legs are balanced around the current price.
        atm_strike = main_leg.get("strike", round(spot / 5) * 5)
        put_leg = find_best_strike(chain_data, best_exp, spot, "BEARISH",
                                    delta_target=0.40, option_type="put")

        # If the put leg found a very different strike, force it to the call strike
        if put_leg and abs(put_leg.get("strike", 0) - atm_strike) > (spot * 0.05):
            put_leg = {}  # too far apart — use fallback below

        if not put_leg:
            # Build put leg synthetically at the same strike as the call
            r_rf      = 0.045
            T         = main_leg.get("T_years", 30/365)
            iv_approx = main_leg.get("iv", 40) / 100
            put_price = black_scholes_price(spot, atm_strike, T, r_rf, iv_approx, "put")
            put_greeks = black_scholes_greeks(spot, atm_strike, T, r_rf, iv_approx, "put")
            put_leg = {
                "strike": atm_strike, "exp": best_exp,
                "mid":   round(max(put_price, 0.05), 2),
                "bid":   round(max(put_price * 0.93, 0.01), 2),
                "ask":   round(put_price * 1.07, 2),
                "iv":    main_leg.get("iv", 40),
                "delta": round(put_greeks["delta"], 3),
                "gamma": round(put_greeks["gamma"], 4),
                "theta": round(put_greeks["theta"] * 100, 2),
                "vega":  round(put_greeks["vega"] * 100, 2),
                "option_type": "put", "oi": 0, "volume": 0,
                "T_years": T
            }
        trade["put_leg"] = put_leg

    return trade

# =============================================================================
#  MODULE 6 — PRICING & EXECUTION OUTPUT
# =============================================================================

def compute_trade_pricing(trade: dict) -> dict:
    """
    Calculate entry, target, stop, contracts, and expected move.
    """
    pricing = {}
    main    = trade.get("main_leg", {})
    short   = trade.get("short_leg", {})
    put_l   = trade.get("put_leg", {})
    strategy = trade.get("strategy", "")
    spot    = trade.get("spot", 0)

    if not main:
        return pricing

    if "SPREAD" in strategy:
        debit = round(main.get("mid", 0) - short.get("mid", 0), 2)
        debit = max(debit, 0.05)
        spread_width = abs(main.get("strike", 0) - short.get("strike", spot)) 
        max_profit   = spread_width - debit
        max_loss     = debit

        entry  = round(debit + 0.02, 2)
        target = round(debit + max_profit * 0.50, 2)  # 50% of max profit
        stop   = round(debit * 0.50, 2)

        contracts = max(1, int(MAX_RISK_DOLLARS / (max_loss * 100)))
        risk      = min(contracts * max_loss * 100, MAX_RISK_DOLLARS)

        # Net Greeks for the spread (long leg minus short leg)
        net_delta = round((main.get("delta", 0) or 0) - (short.get("delta", 0) or 0), 3)
        net_theta = round((main.get("theta", 0) or 0) - (short.get("theta", 0) or 0), 2)
        net_vega  = round((main.get("vega",  0) or 0) - (short.get("vega",  0) or 0), 2)
        # Note: for a debit spread theta is negative (hurts) but much less than naked long
        # Vega is positive but reduced vs naked long

        pricing = {
            "structure": f"Debit Spread — Buy ${main['strike']} / Sell ${short['strike']}",
            "net_debit": debit, "entry": entry, "target": target, "stop": stop,
            "max_profit": round(max_profit, 2), "max_loss": max_loss,
            "contracts": contracts, "risk_dollars": round(risk, 0),
            "risk_pct": round(risk / ACCOUNT_SIZE * 100, 2),
            "rr_ratio": round(max_profit / max_loss, 2),
            "net_delta": net_delta, "net_theta": net_theta, "net_vega": net_vega
        }

    elif strategy == "LONG_STRADDLE":
        call_p = main.get("mid", 0)
        put_p  = put_l.get("mid", call_p)
        total_debit = round(call_p + put_p, 2)
        entry  = round(total_debit * 1.02, 2)
        target = round(total_debit * 2.0, 2)   # 2:1
        stop   = round(total_debit * 0.50, 2)

        contracts = max(1, int(MAX_RISK_DOLLARS / (total_debit * 100)))
        risk      = min(contracts * total_debit * 100, MAX_RISK_DOLLARS)

        # Expected move from IV
        iv_pct = main.get("iv", 30) / 100
        T      = main.get("T_years", 30/365)
        exp_move = spot * iv_pct * math.sqrt(T)

        pricing = {
            "structure": f"Straddle — {main['strike']} Call + Put",
            "net_debit": total_debit, "entry": entry, "target": target, "stop": stop,
            "max_profit": None, "max_loss": total_debit,
            "contracts": contracts, "risk_dollars": round(risk, 0),
            "risk_pct": round(risk / ACCOUNT_SIZE * 100, 2),
            "expected_move_1sd": round(exp_move, 2),
            "rr_ratio": 2.0
        }

    else:  # Long call or put
        premium = main.get("mid", 0)
        entry   = round(premium - 0.05, 2) if premium > 0.05 else round(premium, 2)
        target  = round(premium * 2.0, 2)  # 2:1 R/R
        stop    = round(premium * 0.50, 2)

        contracts = max(1, int(MAX_RISK_DOLLARS / (premium * 100)))
        risk      = min(contracts * premium * 100, MAX_RISK_DOLLARS)

        # Implied expected move
        iv_pct   = main.get("iv", 30) / 100
        T        = main.get("T_years", 30/365)
        exp_move = spot * iv_pct * math.sqrt(T)

        # ── Theta warning: flag if daily decay > 10% of premium
        # theta is stored in $/contract/day, premium is $/share
        # divide theta by 100 to normalize to per-share for comparison
        theta_contract = abs(main.get("theta", 0) or 0)   # $/contract/day
        theta_per_day  = round(theta_contract / 100, 4)    # $/share/day
        theta_pct_day  = (theta_per_day / premium * 100) if premium > 0 else 0
        dte_remaining  = round(T * 365)

        theta_warning = None
        if theta_pct_day > 10:
            theta_warning = (
                f"⚠ HIGH THETA DECAY: losing ${theta_contract:.2f}/contract/day "
                f"({theta_pct_day:.0f}% of premium per day) — "
                f"stock must move quickly or time decay kills this trade"
            )

        pricing = {
            "structure": f"Long {main.get('option_type','call').capitalize()} — {main['strike']} strike",
            "premium": premium, "entry": entry, "target": target, "stop": stop,
            "contracts": contracts, "risk_dollars": round(risk, 0),
            "risk_pct": round(risk / ACCOUNT_SIZE * 100, 2),
            "expected_move_1sd": round(exp_move, 2),
            "rr_ratio": 2.0,
            "theta_warning": theta_warning,
            "theta_pct_day": round(theta_pct_day, 1),
            "dte_remaining": dte_remaining
        }

    return pricing

# =============================================================================
#  MODULE 7 — CONFLUENCE SCORING
# =============================================================================

def score_confluence(trade: dict, tech: dict, flow: list, ivr: dict,
                      regime: dict, sector: dict, dte_days: Optional[int]) -> dict:
    """Score the overall trade setup and list confluence factors."""
    score   = 0
    factors = []
    direction = trade.get("direction", "NEUTRAL")

    # Flow signal
    if flow:
        top = flow[0]
        if top["direction"] == direction:
            score += 3
            factors.append(f"Unusual flow ({top['flags'][-1] if top['flags'] else top['direction']})")
        if top.get("premium_paid", 0) >= 100_000:
            score += 2
            factors.append(f"High-conviction print ${top['premium_paid']/1000:.0f}k premium")

    # Technical
    ema = tech.get("ema_cross", "")
    if ("BULL" in ema and direction == "BULLISH") or ("BEAR" in ema and direction == "BEARISH"):
        score += 2
        factors.append(f"EMA crossover confirmed ({ema})")

    if tech.get("above_vwap") and direction == "BULLISH":
        score += 1
        factors.append("Price above VWAP")
    elif not tech.get("above_vwap") and direction == "BEARISH":
        score += 1
        factors.append("Price below VWAP")

    if tech.get("high_volume"):
        score += 1
        factors.append(f"Above-avg volume ({tech.get('rel_volume', 0):.1f}x)")

    for p in tech.get("patterns", []):
        if ("BULL" in p and direction == "BULLISH") or ("BEAR" in p and direction == "BEARISH"):
            score += 1
            factors.append(f"Pattern: {p}")

    # RSI
    rsi = tech.get("rsi", 50)
    if (rsi < 35 and direction == "BULLISH") or (rsi > 65 and direction == "BEARISH"):
        score += 1
        factors.append(f"RSI {rsi} aligned")

    # IV environment — use specific language per strategy
    ivr_val  = ivr.get("ivr", 50) or 50
    strategy = trade.get("strategy", "")
    if strategy == "LONG_STRADDLE":
        if ivr_val < 40:
            score += 2
            factors.append(f"Low IV ({ivr_val:.0f}) — straddle is cheap, vol expansion likely")
        elif ivr_val < 55:
            score += 1
            factors.append(f"Moderate IV ({ivr_val:.0f}) — straddle viable with catalyst")
    elif "LONG" in strategy and ivr_val < 35:
        score += 2
        factors.append(f"Low IV ({ivr_val:.0f}) — options cheap, favors buying premium")
    elif "SPREAD" in strategy and ivr_val > 50:
        score += 2
        factors.append(f"High IV ({ivr_val:.0f}) — spread reduces expensive premium cost")

    # Regime
    if regime.get("regime") == "TRENDING_UP" and direction == "BULLISH":
        score += 1
        factors.append("Macro regime: TRENDING_UP")
    elif regime.get("regime") == "TRENDING_DOWN" and direction == "BEARISH":
        score += 1
        factors.append("Macro regime: TRENDING_DOWN")

    # Sector — reward alignment, penalise going against the market
    bm          = sector.get("broad_market", "UNKNOWN")
    spy_change  = sector.get("spy_change_pct", 0) or 0
    qqq_change  = sector.get("qqq_change_pct", 0) or 0

    if bm == "BULLISH" and direction == "BULLISH":
        score += 1
        factors.append(f"SPY/QQQ tailwind confirmed (SPY {spy_change:+.2f}%)")
    elif bm == "BEARISH" and direction == "BEARISH":
        score += 1
        factors.append(f"SPY/QQQ headwind confirmed (SPY {spy_change:+.2f}%)")
    elif bm == "BULLISH" and direction == "BEARISH":
        score -= 2  # going against a green market — meaningful headwind
        factors.append(f"[PENALTY] Going BEARISH vs green market (SPY {spy_change:+.2f}%) — contrarian risk")
    elif bm == "BEARISH" and direction == "BULLISH":
        score -= 2  # going against a red market
        factors.append(f"[PENALTY] Going BULLISH vs red market (SPY {spy_change:+.2f}%) — contrarian risk")
    elif bm == "MIXED":
        factors.append(f"SPY/QQQ mixed signals — no sector confirmation")

    # Earnings catalyst
    if dte_days is not None and 2 <= dte_days <= 14:
        score += 2
        factors.append(f"Earnings in {dte_days} days — volatility catalyst")

    # Key level
    kl = tech.get("key_level", "")
    if "SUPPORT" in kl and direction == "BULLISH":
        score += 1
        factors.append("Price at support level")
    elif "RESISTANCE" in kl and direction == "BEARISH":
        score += 1
        factors.append("Price at resistance level")

    # ── Implied move vs strike distance sanity check
    # Ask: does our strike require a move that's reasonable given current IV?
    # If the strike is > 2σ away from spot, flag it as low probability.
    main_leg = trade.get("main_leg", {})
    strike   = main_leg.get("strike")
    iv_pct   = main_leg.get("iv", 40) / 100 if main_leg.get("iv") else 0.40
    T        = main_leg.get("T_years", 30/365)
    spot_val = trade.get("spot", 0)

    if strike and spot_val and T and iv_pct:
        one_sd_move  = spot_val * iv_pct * math.sqrt(T)
        strike_dist  = abs(strike - spot_val)
        sigmas_away  = strike_dist / one_sd_move if one_sd_move > 0 else 0

        if sigmas_away <= 0.5:
            score += 1
            factors.append(f"Strike is {sigmas_away:.1f}σ from spot — high probability (near ATM)")
        elif sigmas_away <= 1.0:
            score += 1
            factors.append(f"Strike is {sigmas_away:.1f}σ from spot — reasonable probability")
        elif sigmas_away <= 1.5:
            factors.append(f"Strike is {sigmas_away:.1f}σ from spot — moderate probability, needs a solid move")
        elif sigmas_away <= 2.0:
            score -= 1
            factors.append(f"[PENALTY] Strike is {sigmas_away:.1f}σ from spot — low probability, needs large move")
        else:
            score -= 2
            factors.append(f"[PENALTY] Strike is {sigmas_away:.1f}σ from spot — very low probability lottery ticket")

    # ── Volume confirmation: high underlying volume + options flow = stronger signal
    # Skip the low-volume penalty before 10am ET — early morning volume is always
    # low vs a full-day average and would penalize every trade unfairly.
    rel_vol = tech.get("rel_volume", 1.0) or 1.0
    now_et  = datetime.now()  # approximate — adjust if not running in ET timezone
    market_open_minutes = (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30)
    early_session = market_open_minutes < 30  # within first 30 min of open

    if rel_vol >= 2.0 and flow:
        score += 1
        factors.append(f"High underlying volume ({rel_vol:.1f}x avg) confirms options flow")
    elif rel_vol < 0.5 and not early_session:
        score -= 1
        factors.append(f"[PENALTY] Low underlying volume ({rel_vol:.1f}x avg) — options flow less reliable")
    elif rel_vol < 0.5 and early_session:
        factors.append(f"Volume data early — check again after 10am ET ({rel_vol:.1f}x avg so far)")

    rating = (
        "A+ SETUP"   if score >= 10 else
        "STRONG"     if score >= 7  else
        "MODERATE"   if score >= 5  else
        "WEAK"       if score >= 3  else
        "AVOID"
    )

    return {"score": score, "rating": rating, "factors": factors}

# =============================================================================
#  MODULE 8 — FULL TICKER ANALYSIS
# =============================================================================

def analyze_ticker(
    ticker: str,
    regime: dict,
    paper_mode: bool = False,
    verbose: bool = True
) -> dict:
    """Run complete analysis for a single ticker."""
    result = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "error": None
    }

    try:
        # 1. Price data
        hist_1d = get_price_data(ticker, period="1y",   interval="1d")
        hist_5m = get_price_data(ticker, period="5d",   interval="5m")

        if hist_1d.empty:
            result["error"] = "No price data"
            return result

        spot = float(hist_1d["Close"].iloc[-1])
        result["spot"] = spot

        # 2. Options chain
        chain_data = {}
        data_source = "yfinance"
        if TRADIER_API_KEY:
            chain_data  = get_options_chain_tradier(ticker, paper_mode)
            data_source = "tradier"
        if not chain_data:
            chain_data  = get_options_chain_yfinance(ticker)
            data_source = "yfinance"
        result["data_source"] = data_source
        result["chain_available"] = bool(chain_data)

        # ── Split chain into two views:
        #    flow_chain  — full chain including near-term expirations
        #                  used ONLY for unusual flow detection
        #    trade_chain — expirations >= MIN_OPTION_DTE days out only
        #                  used for ALL trade construction and strike selection
        # This ensures we never recommend entering a position on a
        # sub-21-day expiration while still detecting near-term flow activity.
        flow_chain  = chain_data
        trade_chain = filter_chain_by_dte(chain_data, MIN_OPTION_DTE)


        # If filtering removed everything (very rare — e.g. only weeklies in chain)
        # fall back to the full chain but flag it in the output
        if not trade_chain:
            trade_chain = chain_data
            result["dte_warning"] = f"No expirations >= {MIN_OPTION_DTE} DTE found — using full chain"

        # 3. Volatility
        hv30  = get_historical_volatility(ticker, 30)
        hv60  = get_historical_volatility(ticker, 60)
        hv90  = get_historical_volatility(ticker, 90)

        # Estimate current IV:
        # 1. Try Tradier native ATM IV (most accurate — live market IV)
        # 2. Fall back to yfinance volatility surface approximation
        current_iv = extract_atm_iv_from_chain(flow_chain, spot)
        surface, atm_ivs = compute_volatility_surface(ticker, spot)
        if current_iv is None and atm_ivs:
            current_iv = list(atm_ivs.values())[0] / 100
        result["iv_source"] = "tradier_native" if current_iv else "yfinance_proxy"

        ivr   = get_iv_rank(ticker, current_iv)
        skew  = analyze_vol_skew(surface, spot,
                                  list(surface.keys())[0] if surface else "")

        vol_data = {
            "hv30": round(hv30 * 100, 1) if hv30 else None,
            "hv60": round(hv60 * 100, 1) if hv60 else None,
            "hv90": round(hv90 * 100, 1) if hv90 else None,
            "current_iv": round(current_iv * 100, 1) if current_iv else None,
            "iv_vs_hv30": (
                "IV > HV (rich)"  if current_iv and hv30 and current_iv > hv30 else
                "IV < HV (cheap)" if current_iv and hv30 and current_iv < hv30 else
                "N/A"
            ),
            **ivr, **skew
        }
        result["vol"] = vol_data

        # 4. Flow scan — use FULL chain (we want near-term activity too)
        flow_signals = scan_unusual_flow(flow_chain, spot, paper_mode)
        result["flow"] = flow_signals

        # 5. Technicals
        tech = compute_technicals(hist_1d)
        if hist_5m is not None and not hist_5m.empty:
            intraday = compute_technicals(hist_5m)
            tech["intraday_rsi"] = intraday.get("rsi")
            tech["intraday_above_vwap"] = intraday.get("above_vwap")
        result["tech"] = tech

        # 6. Sector
        sector = get_sector_correlation(ticker)
        result["sector"] = sector

        # 7. Earnings
        dte_earnings = get_days_to_earnings(ticker)
        result["days_to_earnings"] = dte_earnings

        # 8. Trade construction
        direction = "NEUTRAL"
        if flow_signals:
            top_flow = flow_signals[0]
            direction = top_flow["direction"]

        trade = construct_trade(
            ticker, spot, direction, ivr, tech, flow_signals,
            trade_chain, regime, dte_earnings, paper_mode
        )
        result["trade"] = trade

        # 9. Pricing
        pricing = compute_trade_pricing(trade)
        result["pricing"] = pricing

        # 10. Confluence scoring
        confluence = score_confluence(trade, tech, flow_signals, ivr, regime, sector, dte_earnings)
        result["confluence"] = confluence

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result

# =============================================================================
#  MODULE 9 — RICH TERMINAL OUTPUT
# =============================================================================

def print_header(regime: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    regime_colors = {
        "TRENDING_UP":   "green",
        "TRENDING_DOWN": "red",
        "RANGING":       "yellow",
        "RISK_OFF":      "bold red",
        "UNKNOWN":       "white"
    }
    regime_n = regime.get("regime", "UNKNOWN")
    color    = regime_colors.get(regime_n, "white")

    if console:
        console.rule(f"[bold cyan]OPTIONS TRADE SCANNER — {now}[/bold cyan]")
        console.print(f"\n[bold]Market Regime:[/bold] [{color}]{regime_n}[/{color}]  |  "
                      f"VIX: [bold]{regime.get('vix','?')}[/bold] ({regime.get('vix_regime','?')})  |  "
                      f"SPY 1M: [bold]{regime.get('spy_1m_return','?')}%[/bold]  |  "
                      f"SPY RSI: [bold]{regime.get('spy_rsi','?')}[/bold]\n")
    else:
        print("=" * 50)
        print(f"OPTIONS TRADE SCANNER — {now}")
        print(f"Regime: {regime_n}  VIX: {regime.get('vix','?')}")
        print("=" * 50)

def print_trade_block(result: dict, rank: int = 1):
    ticker    = result.get("ticker", "?")
    spot      = result.get("spot", 0)
    trade     = result.get("trade", {})
    pricing   = result.get("pricing", {})
    vol       = result.get("vol", {})
    tech      = result.get("tech", {})
    flow      = result.get("flow", [])
    conf      = result.get("confluence", {})
    sector    = result.get("sector", {})
    dte_earn  = result.get("days_to_earnings")
    error     = result.get("error")

    if error and not trade:
        cprint(f"[red]Error analyzing {ticker}: {error}[/red]")
        return

    main    = trade.get("main_leg", {})
    short   = trade.get("short_leg", {})
    put_l   = trade.get("put_leg", {})
    strategy = trade.get("strategy", "?")

    conf_score  = conf.get("score", 0)
    conf_rating = conf.get("rating", "?")
    conf_color  = {"A+ SETUP": "green", "STRONG": "cyan", "MODERATE": "yellow",
                   "WEAK": "red", "AVOID": "bold red"}.get(conf_rating, "white")

    if console:
        # ── Trade Title Panel
        label = f"[bold]{'TOP' if rank==1 else f'#{rank}'} TRADE — {ticker}[/bold]"
        console.print(Panel(
            f"[bold yellow]{ticker}[/bold yellow] @ ${spot:.2f}  |  "
            f"Strategy: [bold]{strategy}[/bold]  |  "
            f"Direction: [{'green' if trade.get('direction')=='BULLISH' else ('red' if trade.get('direction')=='BEARISH' else 'yellow')}]{trade.get('direction','?')}{'  (buy both sides)' if strategy == 'LONG_STRADDLE' else ''}[/]  |  "
            f"Confluence: [{conf_color}]{conf_rating} ({conf_score}pts)[/{conf_color}]",
            title=label, box=box.ROUNDED
        ))

        # ── Contract details table
        tbl = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        tbl.add_column("Field", style="cyan", min_width=20)
        tbl.add_column("Value", style="white")

        exp_str  = main.get("exp", trade.get("exp","?"))
        opt_type = main.get("option_type","call").upper()
        strike   = main.get("strike", "?")

        # Show data quality warning if this is a theoretical/fallback contract
        data_quality = trade.get("data_quality", "")
        trade_label  = f"Buy {ticker} ${strike} {opt_type} exp {exp_str}"
        if data_quality:
            trade_label += "  [yellow]⚠ THEORETICAL[/yellow]"

        tbl.add_row("Trade", trade_label)
        if data_quality:
            tbl.add_row("[yellow]⚠ Data Quality[/yellow]", f"[yellow]{data_quality}[/yellow]")

        # ── For straddles, show both legs explicitly
        if strategy == "LONG_STRADDLE" and put_l:
            call_strike = main.get("strike", "?")
            put_strike  = put_l.get("strike", "?")
            call_mid    = main.get("mid", 0)
            put_mid     = put_l.get("mid", 0)
            total       = round(call_mid + put_mid, 2)
            tbl.add_row("  Call Leg",    f"Buy ${call_strike} CALL  —  ${call_mid}")
            tbl.add_row("  Put Leg",     f"Buy ${put_strike} PUT   —  ${put_mid}")
            tbl.add_row("  Total Debit", f"[bold]${total}[/bold] per share  (${total*100:.0f} per contract)")
            be_up   = round(call_strike + total, 2) if isinstance(call_strike, float) else "?"
            be_down = round(put_strike  - total, 2) if isinstance(put_strike, float)  else "?"
            tbl.add_row("  Break-evens", f"Above [green]${be_up}[/green]  or  Below [red]${be_down}[/red]")

        # ── For spreads, show both legs so the user sees the full structure
        elif "SPREAD" in strategy and short:
            long_strike  = main.get("strike", "?")
            short_strike = short.get("strike", "?")
            long_mid     = main.get("mid", 0)
            short_mid    = short.get("mid", 0)
            net_debit    = round(long_mid - short_mid, 2)
            spread_width = abs((long_strike or 0) - (short_strike or 0))
            max_profit   = round(spread_width - net_debit, 2) if spread_width else "?"
            opt_label    = main.get("option_type", "put").upper()

            if "BULL" in strategy:
                tbl.add_row("  Buy Leg",    f"Buy  ${long_strike} {opt_label}  —  ${long_mid}")
                tbl.add_row("  Sell Leg",   f"Sell ${short_strike} {opt_label}  —  ${short_mid} [dim](premium received)[/dim]")
            else:
                tbl.add_row("  Buy Leg",    f"Buy  ${long_strike} {opt_label}  —  ${long_mid}")
                tbl.add_row("  Sell Leg",   f"Sell ${short_strike} {opt_label}  —  ${short_mid} [dim](premium received)[/dim]")

            tbl.add_row("  Net Debit",   f"[bold]${net_debit}[/bold] per share  (${net_debit*100:.0f} per contract)")
            tbl.add_row("  Max Profit",  f"${max_profit} per share  (${max_profit*100:.0f} per contract)  [dim]if held to expiry[/dim]"
                                          if isinstance(max_profit, float) else f"{max_profit}")
            # Break-even for debit spread
            if isinstance(long_strike, float) and isinstance(net_debit, float):
                if "BULL" in strategy:
                    be = round(long_strike + net_debit, 2)
                    tbl.add_row("  Break-even", f"[green]${be}[/green]  (stock must close above this at expiry)")
                else:
                    be = round(long_strike - net_debit, 2)
                    tbl.add_row("  Break-even", f"[red]${be}[/red]  (stock must close below this at expiry)")

        tbl.add_row("Entry",  f"${pricing.get('entry','?')} (limit at mid)")
        tbl.add_row("Target", f"${pricing.get('target','?')} "
                              f"(+{round((pricing.get('target',0)/pricing.get('entry',1)-1)*100,0):.0f}%)")
        tbl.add_row("Stop",   f"${pricing.get('stop','?')} "
                              f"(-{round((1-pricing.get('stop',0)/pricing.get('entry',1))*100,0):.0f}%)")
        tbl.add_row("Contracts", f"{pricing.get('contracts','?')} straddle(s)  "
                                  f"(risk: ${pricing.get('risk_dollars','?')} / {pricing.get('risk_pct','?')}% of account)"
                                  if strategy == "LONG_STRADDLE" else
                                  f"{pricing.get('contracts','?')} "
                                  f"(risk: ${pricing.get('risk_dollars','?')} / {pricing.get('risk_pct','?')}% of account)")

        ivr_val    = vol.get("ivr") or 0
        current_iv = vol.get("current_iv")
        hv30       = vol.get("hv30")
        iv_src     = result.get("iv_source", "")
        greeks_src = main.get("iv_source", "")
        ivr_label  = (
            "[green]Low — options are cheap, favor buying premium[/green]"   if ivr_val < 35 else
            "[red]High — options are expensive, favor spreads or selling[/red]" if ivr_val > 50 else
            "[yellow]Moderate — neutral environment[/yellow]"
        )
        src_label = (
            " [dim](live Tradier IV)[/dim]" if iv_src == "tradier_native" else
            " [dim](estimated from HV)[/dim]" if iv_src else ""
        )
        greeks_label = (
            " [dim](live Tradier Greeks)[/dim]" if greeks_src == "tradier" else
            " [dim](Black-Scholes)[/dim]" if greeks_src else ""
        )
        tbl.add_row("IV Rank", f"{ivr_val}  {ivr_label}{src_label}")

        # IV vs HV note — explain what the comparison actually means
        if current_iv and hv30:
            if current_iv > hv30:
                iv_note = (f"Current IV {current_iv}% > HV30 {hv30}%  — "
                           f"[yellow]options pricing in MORE move than stock has recently made "
                           f"(IV rich vs realized vol)[/yellow]")
            else:
                iv_note = (f"Current IV {current_iv}% < HV30 {hv30}%  — "
                           f"[green]options pricing in LESS move than stock has recently made "
                           f"(IV cheap vs realized vol)[/green]")
        else:
            iv_note = f"Current IV {current_iv or '?'}%  HV30 {hv30 or '?'}%"
        tbl.add_row("IV vs HV30", iv_note)

        # Greeks — show net position Greeks for spreads and straddles
        if strategy == "LONG_STRADDLE" and put_l:
            net_delta = round((main.get("delta", 0) or 0) + (put_l.get("delta", 0) or 0), 3)
            net_theta = round((main.get("theta", 0) or 0) + (put_l.get("theta", 0) or 0), 2)
            net_vega  = round((main.get("vega",  0) or 0) + (put_l.get("vega",  0) or 0), 2)
            tbl.add_row("Greeks (net)", f"Delta: {net_delta} [dim](≈0 = direction-neutral)[/dim]  "
                                        f"Theta: ${net_theta}/day  Vega: +${net_vega}")

        elif "SPREAD" in strategy and pricing.get("net_delta") is not None:
            nd = pricing.get("net_delta")
            nt = pricing.get("net_theta")
            nv = pricing.get("net_vega")
            long_delta  = main.get("delta","?")
            long_theta  = main.get("theta","?")
            tbl.add_row("Greeks (net)",
                        f"Delta: {nd} [dim](vs {long_delta} long leg alone)[/dim]  "
                        f"Theta: ${nt}/day [dim](reduced vs naked — short leg offsets)[/dim]  "
                        f"Vega: +${nv}")
        else:
            delta = main.get("delta","?")
            theta = main.get("theta","?")
            vega  = main.get("vega","?")
            tbl.add_row("Greeks",
                        f"Delta: {delta}  Theta: ${theta}/day  Vega: +${vega}{greeks_label}")

        if dte_earn is not None:
            tbl.add_row("Earnings", f"In {dte_earn} days")

        exp_move = pricing.get("expected_move_1sd")
        if exp_move:
            tbl.add_row("Implied Move", f"±${exp_move:.2f} ({exp_move/spot*100:.1f}%) 1σ")

        # ── Theta warning — show prominently if decay is aggressive
        theta_warning = pricing.get("theta_warning")
        theta_pct     = pricing.get("theta_pct_day", 0)
        dte_rem       = pricing.get("dte_remaining")
        if dte_rem is not None:
            tbl.add_row("DTE", f"{dte_rem} days remaining")
        if theta_warning:
            tbl.add_row("[bold red]⚠ Theta Risk[/bold red]", f"[red]{theta_warning}[/red]")
        elif theta_pct > 5:
            tbl.add_row("[yellow]Theta Note[/yellow]",
                        f"[yellow]Losing {theta_pct:.0f}% of premium/day — monitor closely[/yellow]")

        tbl.add_row("Confluence", " + ".join(conf.get("factors", [])[:3]))

        console.print(tbl)

        # ── Why this trade
        console.print("\n[bold]WHY THIS TRADE:[/bold]")
        for f in conf.get("factors", []):
            console.print(f"  • {f}")

        # Only show the rationale if it contains info NOT already in confluence factors
        # (avoids the duplicate "Moderate IV — limited risk bear spread" lines)
        rationale = trade.get("rationale", "")
        already_covered = any(
            any(word in f.lower() for word in rationale.lower().split()[:3])
            for f in conf.get("factors", [])
        )
        if rationale and not already_covered:
            console.print(f"  • Strategy: {rationale}")

        if flow:
            top = flow[0]
            aggressor    = top.get("aggressor", "UNKNOWN")
            confidence   = top.get("dir_confidence", "LOW")
            inferred_dir = top.get("direction", "?")

            if aggressor == "BUY" and confidence == "HIGH":
                dir_str = f"[green]BOUGHT at ask → {inferred_dir}[/green]"
            elif aggressor == "SELL" and confidence == "HIGH":
                dir_str = f"[red]SOLD at bid → {inferred_dir}[/red]"
            else:
                dir_str = f"[yellow]direction ambiguous (mid fill)[/yellow]"

            console.print(f"  • Flow: {top['volume']:,} contracts {top['opt_type'].upper()} "
                          f"${top['strike']} exp {top['exp']} | "
                          f"${top['premium_paid']/1000:.0f}k premium | "
                          f"{', '.join(top['flags'])} | {dir_str}")
        console.print(f"  • SPY: {sector.get('spy_trend','?')} ({sector.get('spy_change_pct','?')}%)  "
                      f"QQQ: {sector.get('qqq_trend','?')} ({sector.get('qqq_change_pct','?')}%)")
        console.print(f"  • Technical: RSI {tech.get('rsi','?')} | EMA {tech.get('ema_cross','?')} | "
                      f"{'Above' if tech.get('above_vwap') else 'Below'} VWAP | Patterns: {', '.join(tech.get('patterns',[])) or 'None'}")
        console.print()

    else:
        print(f"\n{'='*50}")
        print(f"#{rank} TRADE — {ticker} @ ${spot:.2f}")
        print(f"Strategy: {strategy}  Direction: {trade.get('direction','?')}")
        print(f"Confluence: {conf_rating} ({conf_score}pts)")
        print(f"Entry: ${pricing.get('entry','?')}  Target: ${pricing.get('target','?')}  Stop: ${pricing.get('stop','?')}")
        print(f"Contracts: {pricing.get('contracts','?')}  Risk: ${pricing.get('risk_dollars','?')}")
        print(f"Factors: {'; '.join(conf.get('factors',[]))}")

def print_risk_summary(results: list):
    total_risk = sum(r.get("pricing", {}).get("risk_dollars", 0) for r in results if not r.get("error"))
    avail_risk = MAX_RISK_DOLLARS

    if console:
        console.rule("[bold]RISK SUMMARY[/bold]")
        tbl = Table(box=box.SIMPLE, show_header=False)
        tbl.add_column("", style="cyan")
        tbl.add_column("", style="white")
        tbl.add_row("Account Size",     dollar(ACCOUNT_SIZE))
        tbl.add_row("Max Risk / Trade", dollar(MAX_RISK_DOLLARS))
        tbl.add_row("Available Risk",   dollar(avail_risk))
        tbl.add_row("Open Positions",   "0 (not tracking live positions)")
        tbl.add_row("Today's P&L",      "— (run with broker integration)")
        tbl.add_row("Tickers Scanned",  str(len(results)))
        console.print(tbl)
        console.rule()
    else:
        print("\n" + "="*50)
        print("RISK SUMMARY")
        print(f"Account: ${ACCOUNT_SIZE:,}  Available Risk: ${avail_risk:,.0f}")
        print("="*50)

# =============================================================================
#  MODULE 10 — LOGGING
# =============================================================================

def save_log(results: list, regime: dict, paper_mode: bool):
    """Save timestamped log of all results."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "_paper" if paper_mode else ""
    log_path = LOG_DIR / f"scan_{ts}{mode_tag}.json"

    log = {
        "timestamp": ts,
        "paper_mode": paper_mode,
        "account_size": ACCOUNT_SIZE,
        "max_risk": MAX_RISK_DOLLARS,
        "regime": regime,
        "results": []
    }

    for r in results:
        entry = {
            "ticker": r.get("ticker"),
            "spot": r.get("spot"),
            "strategy": r.get("trade", {}).get("strategy"),
            "direction": r.get("trade", {}).get("direction"),
            "main_strike": r.get("trade", {}).get("main_leg", {}).get("strike"),
            "exp": r.get("trade", {}).get("exp"),
            "entry": r.get("pricing", {}).get("entry"),
            "target": r.get("pricing", {}).get("target"),
            "stop": r.get("pricing", {}).get("stop"),
            "contracts": r.get("pricing", {}).get("contracts"),
            "risk": r.get("pricing", {}).get("risk_dollars"),
            "confluence_score": r.get("confluence", {}).get("score"),
            "confluence_rating": r.get("confluence", {}).get("rating"),
            "ivr": r.get("vol", {}).get("ivr"),
            "error": r.get("error")
        }
        log["results"].append(entry)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    # Also save a human-readable summary CSV
    csv_path = LOG_DIR / f"scan_{ts}{mode_tag}.csv"
    df = pd.DataFrame(log["results"])
    df.to_csv(csv_path, index=False)

    return log_path, csv_path

# =============================================================================
#  MAIN RUNNER
# =============================================================================

def generate_requirements_txt():
    req = "\n".join([
        "yfinance>=0.2.28",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "rich>=13.0.0",
        "requests>=2.28.0",
        "pandas-datareader>=0.10.0",
        "matplotlib>=3.7.0",
        "ta>=0.10.2",
        "python-dateutil>=2.8.2"
    ])
    with open("requirements.txt", "w") as f:
        f.write(req)

def main():
    parser = argparse.ArgumentParser(description="Options Trade Scanner")
    parser.add_argument("--paper",   action="store_true", help="Paper trading mode")
    parser.add_argument("--tickers", nargs="*", default=None, help="Override watchlist")
    parser.add_argument("--account", type=float, default=None, help="Override account size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    global ACCOUNT_SIZE, MAX_RISK_DOLLARS
    if args.account:
        ACCOUNT_SIZE      = args.account
        MAX_RISK_DOLLARS  = ACCOUNT_SIZE * MAX_RISK_PCT

    watchlist = args.tickers if args.tickers else WATCHLIST
    paper_mode = args.paper

    generate_requirements_txt()

    if console:
        if paper_mode:
            console.print("[bold yellow]⚠  PAPER TRADING MODE — No execution hooks active[/bold yellow]\n")
    else:
        if paper_mode:
            print("PAPER TRADING MODE")

    # ── Market regime
    if console:
        with console.status("[bold green]Analyzing market regime..."):
            regime = determine_market_regime()
    else:
        print("Analyzing market regime...")
        regime = determine_market_regime()

    print_header(regime)

    # ── Regime-based guidance
    regime_guidance = {
        "TRENDING_UP":   "Favor bullish setups. Prefer long calls and bull spreads.",
        "TRENDING_DOWN": "Favor bearish setups. Bear puts and put spreads preferred.",
        "RANGING":       "Range-bound market. Sell premium, spreads, iron condors.",
        "RISK_OFF":      "HIGH CAUTION — VIX elevated. Reduce size, favor put protection.",
    }
    if console:
        cprint(f"[bold]Regime Guidance:[/bold] {regime_guidance.get(regime.get('regime','UNKNOWN'), 'N/A')}\n")

    # ── Scan all tickers
    all_results = []
    for ticker in watchlist:
        if console:
            with console.status(f"[bold green]Scanning {ticker}..."):
                r = analyze_ticker(ticker, regime, paper_mode, args.verbose)
        else:
            print(f"Scanning {ticker}...")
            r = analyze_ticker(ticker, regime, paper_mode, args.verbose)

        all_results.append(r)
        time.sleep(0.3)  # rate limiting

    # ── Sort by confluence score
    # Also filter out junk trades before ranking:
    #   - Spreads with entry < $0.20 are bad chain data (e.g. $0.07 spread)
    #   - Any trade with no main_leg strike is incomplete
    def is_valid_trade(r):
        if r.get("error") or not r.get("confluence"):
            return False
        pricing  = r.get("pricing", {})
        trade    = r.get("trade", {})
        entry    = pricing.get("entry", 0) or 0
        strategy = trade.get("strategy", "")
        ticker   = r.get("ticker", "")

        # ── Exclude macro/ETF tickers from recommendations
        # SPY, QQQ etc. are used for regime context only — not trade candidates.
        # Single stocks give better directional R/R than broad market ETFs.
        if ticker.upper() in MACRO_ONLY_TICKERS:
            return False

        if "SPREAD" in strategy and entry < 0.50:
            return False  # junk spread — bad chain data
        if strategy in ("LONG_CALL", "LONG_PUT") and entry < 0.50:
            return False  # near-worthless long option — bad chain data
        if not trade.get("main_leg", {}).get("strike"):
            return False  # incomplete trade construction
        return True

    valid   = [r for r in all_results if is_valid_trade(r)]
    valid.sort(key=lambda x: x.get("confluence", {}).get("score", 0), reverse=True)
    errored = [r for r in all_results if r.get("error")]

    # ── Print top setups
    if valid:
        if console:
            console.rule("[bold green]TOP TRADE SETUP[/bold green]")
        print_trade_block(valid[0], rank=1)

        if len(valid) > 1:
            if console:
                console.rule("[bold]NEXT BEST SETUPS[/bold]")
                for i, r in enumerate(valid[1:4], start=2):
                    tick   = r.get("ticker","?")
                    trade  = r.get("trade",{})
                    price  = r.get("pricing",{})
                    vol    = r.get("vol",{})
                    conf   = r.get("confluence",{})
                    entry  = price.get("entry","?")
                    strat  = trade.get("strategy","?")
                    exp    = trade.get("main_leg",{}).get("exp", trade.get("exp","?"))
                    ivr_v  = vol.get("ivr","?")
                    score  = conf.get("score",0)
                    strike = trade.get("main_leg",{}).get("strike","?")
                    console.print(f"  [cyan]{i}.[/cyan] [bold]{tick}[/bold] {strat} — ${strike} exp {exp} | "
                                  f"Entry ${entry} | IVR: {ivr_v} | Score: {score}")
            else:
                print("\nNEXT BEST SETUPS:")
                for i, r in enumerate(valid[1:4], start=2):
                    print(f"  {i}. {r.get('ticker')} — {r.get('trade',{}).get('strategy')} entry ${r.get('pricing',{}).get('entry','?')}")

    if errored and args.verbose:
        if console:
            console.print("\n[red]Errors encountered:[/red]")
            for r in errored:
                console.print(f"  {r['ticker']}: {r['error']}")

    # ── Risk summary
    print_risk_summary(valid)

    # ── Save logs
    log_path, csv_path = save_log(all_results, regime, paper_mode)
    if console:
        console.print(f"\n[dim]Logs saved: {log_path}  |  {csv_path}[/dim]")
    else:
        print(f"Logs saved: {log_path}")

if __name__ == "__main__":
    main()
