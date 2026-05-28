"""
=============================================================================
broker.py — Alpaca Paper Trading Execution Layer
=============================================================================
Sends real orders to Alpaca paper trading account.
Uses Tradier production key for market data (real-time).
Uses Alpaca paper key for order execution with web dashboard visibility.

Dashboard: https://app.alpaca.markets/paper-trading/overview

Usage:
  python3 rl_system/run.py --live-paper --bankroll 5000
  python3 rl_system/run.py --live-paper --auto

Environment variables required:
  ALPACA_API_KEY     — paper trading key ID
  ALPACA_SECRET_KEY  — paper trading secret key
  TRADIER_API_KEY    — production key for real-time market data
=============================================================================
"""

import os
import time
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# ─── Endpoints ───────────────────────────────────────────────────────────────
ALPACA_BASE      = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"

# ─── Keys (from environment) ─────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")


def _headers() -> Dict:
    return {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type":        "application/json",
        "Accept":              "application/json"
    }


def _is_configured() -> bool:
    return bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)


def get_account_balance() -> Optional[float]:
    """Return available cash balance from Alpaca paper account."""
    if not _is_configured():
        return None
    try:
        r = requests.get(f"{ALPACA_BASE}/account", headers=_headers(), timeout=8)
        if r.status_code == 200:
            data = r.json()
            return float(data.get("cash", 0) or 0)
    except Exception as e:
        logger.debug(f"Balance fetch failed: {e}")
    return None


def get_account_info() -> Optional[Dict]:
    """Return full account info dict."""
    if not _is_configured():
        return None
    try:
        r = requests.get(f"{ALPACA_BASE}/account", headers=_headers(), timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug(f"Account info fetch failed: {e}")
    return None


def build_option_symbol(ticker: str, expiration: str, option_type: str,
                        strike: float) -> str:
    """
    Build OCC option symbol for Alpaca.
    Format: TICKER + YYMMDD + C/P + 8-digit strike (strike * 1000, zero-padded)

    Examples:
        GOOGL, 2026-05-15, call, 340.0  → GOOGL260515C00340000
        AAPL,  2026-05-15, put,  270.0  → AAPL260515P00270000
        MU,    2026-05-15, call, 187.5  → MU260515C00187500
    """
    try:
        exp_dt     = datetime.strptime(expiration, "%Y-%m-%d")
        exp_str    = exp_dt.strftime("%y%m%d")
        cp         = "C" if option_type.lower() == "call" else "P"
        strike_int = int(round(strike * 1000))
        strike_str = str(strike_int).zfill(8)
        return f"{ticker}{exp_str}{cp}{strike_str}"
    except Exception as e:
        logger.error(f"Symbol build failed: {e}")
        return ""


def place_option_order(
    ticker: str,
    expiration: str,
    option_type: str,
    strike: float,
    side: str,           # "buy" or "sell"
    quantity: int,
    limit_price: float,
    tag: str = "",
    market_order: bool = False
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Place a single-leg option order on Alpaca paper account.
    Returns: (success, order_id, fill_price)
    """
    if not _is_configured():
        logger.error("Alpaca not configured — set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return False, None, None

    symbol = build_option_symbol(ticker, expiration, option_type, strike)
    if not symbol:
        return False, None, None

    # Alpaca single-leg options
    # For closing orders, must include position_intent="sell_to_close"
    # so Alpaca knows it's closing an existing long, not opening a naked short
    if side in ("buy", "buy_to_open"):
        alpaca_side     = "buy"
        position_intent = "buy_to_open"
    else:
        alpaca_side     = "sell"
        position_intent = "sell_to_close"

    if market_order:
        order_type_fields = {"type": "market"}
    else:
        order_type_fields = {"type": "limit", "limit_price": str(round(limit_price, 2))}

    payload = {
        "symbol":          symbol,
        "qty":             str(quantity),
        "side":            alpaca_side,
        **order_type_fields,
        "time_in_force":   "day",
        "position_intent": position_intent,
        "client_order_id": tag or f"rl_{ticker}_{datetime.now().strftime('%H%M%S')}"
    }

    try:
        logger.info(f"Sending order payload: {payload}")
        r = requests.post(
            f"{ALPACA_BASE}/orders",
            headers=_headers(),
            json=payload,
            timeout=10
        )
        if r.status_code in (200, 201):
            data     = r.json()
            order_id = data.get("id", "")
            logger.info(f"Order placed: {side} {quantity}x {symbol} @ ${limit_price} — ID: {order_id}")
            return True, order_id, None
        else:
            logger.error(f"Order failed {r.status_code}: {r.text[:300]}")
            return False, None, None
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        return False, None, None


def place_spread_order(
    ticker: str,
    expiration: str,
    option_type: str,
    long_strike: float,
    short_strike: float,
    side: str,           # "buy" to open, "sell" to close
    quantity: int,
    net_limit_price: float,
    tag: str = "",
    market_order: bool = False
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Place a two-leg spread order on Alpaca paper account.
    Alpaca supports multi-leg orders via the orders endpoint with legs array.
    """
    if not _is_configured():
        return False, None, None

    long_symbol  = build_option_symbol(ticker, expiration, option_type, long_strike)
    short_symbol = build_option_symbol(ticker, expiration, option_type, short_strike)

    if not long_symbol or not short_symbol:
        return False, None, None

    long_side  = "buy"  if side == "buy" else "sell"
    short_side = "sell" if side == "buy" else "buy"

    # position_intent for each leg based on whether opening or closing.
    # Opening a debit spread: buy long leg to open, sell short leg to open.
    # Closing it: sell long leg to close, buy short leg to close.
    long_intent  = "buy_to_open"   if side == "buy" else "sell_to_close"
    short_intent = "sell_to_open"  if side == "buy" else "buy_to_close"

    if market_order:
        order_type_fields = {"type": "market"}
    else:
        order_type_fields = {"type": "limit", "limit_price": str(round(net_limit_price, 2))}

    payload = {
        **order_type_fields,
        "time_in_force": "day",
        "order_class":   "mleg",
        "qty":           str(quantity),   # required at parent level for mleg
        "legs": [
            {
                "symbol":          long_symbol,
                "side":            long_side,
                "ratio_qty":       "1",
                "position_intent": long_intent
            },
            {
                "symbol":          short_symbol,
                "side":            short_side,
                "ratio_qty":       "1",
                "position_intent": short_intent
            }
        ],
        "client_order_id": tag or f"rl_{ticker}_spread_{datetime.now().strftime('%H%M%S')}"
    }

    try:
        r = requests.post(
            f"{ALPACA_BASE}/orders",
            headers=_headers(),
            json=payload,
            timeout=10
        )
        if r.status_code in (200, 201):
            data     = r.json()
            order_id = data.get("id", "")
            logger.info(f"Spread order placed: {long_symbol}/{short_symbol} @ ${net_limit_price} net — ID: {order_id}")
            return True, order_id, None
        else:
            logger.error(f"Spread order failed {r.status_code}: {r.text[:300]}")
            return False, None, None
    except Exception as e:
        logger.error(f"Spread order error: {e}")
        return False, None, None


def poll_for_fill(order_id: str, timeout_seconds: int = 60,
                  poll_interval: int = 5) -> Tuple[str, Optional[float]]:
    """
    Poll Alpaca until order fills, cancels, or times out.
    Returns: (status, fill_price)
    """
    if not _is_configured() or not order_id:
        return "error", None

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{ALPACA_BASE}/orders/{order_id}",
                headers=_headers(),
                timeout=8
            )
            if r.status_code == 200:
                order  = r.json()
                status = order.get("status", "").lower()

                if status == "filled":
                    fill_price = float(order.get("filled_avg_price", 0) or 0)
                    logger.info(f"Order {order_id} filled @ ${fill_price:.2f}")
                    return "filled", fill_price

                if status in ("canceled", "cancelled", "rejected", "expired"):
                    logger.warning(f"Order {order_id} {status}")
                    return status, None

                logger.debug(f"Order {order_id} status: {status} — waiting...")
            else:
                logger.debug(f"Poll error {r.status_code}")

        except Exception as e:
            logger.debug(f"Poll error: {e}")

        time.sleep(poll_interval)

    logger.warning(f"Order {order_id} timed out after {timeout_seconds}s")
    return "timeout", None


def cancel_order(order_id: str) -> bool:
    """Cancel an open order."""
    if not _is_configured() or not order_id:
        return False
    try:
        r = requests.delete(
            f"{ALPACA_BASE}/orders/{order_id}",
            headers=_headers(),
            timeout=8
        )
        return r.status_code in (200, 204)
    except Exception as e:
        logger.debug(f"Cancel error: {e}")
        return False


def get_positions() -> list:
    """Return all open positions from Alpaca paper account."""
    if not _is_configured():
        return []
    try:
        r = requests.get(f"{ALPACA_BASE}/positions", headers=_headers(), timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug(f"Positions fetch failed: {e}")
    return []


def get_open_orders() -> list:
    """Return all open orders from Alpaca paper account."""
    if not _is_configured():
        return []
    try:
        r = requests.get(
            f"{ALPACA_BASE}/orders",
            headers=_headers(),
            params={"status": "open"},
            timeout=8
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug(f"Open orders fetch failed: {e}")
    return []
