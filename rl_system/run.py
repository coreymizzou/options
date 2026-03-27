"""
=============================================================================
run.py — Main Orchestration Loop
=============================================================================
Runs the 60-second decision loop:
  1. Every SCANNER_RUN_INTERVAL seconds: run full options scanner
  2. Every 60 seconds: evaluate open positions + new candidates
  3. Agent scores ENTER / HOLD / EXIT for each
  4. Hard risk rules checked before any recommendation is surfaced
  5. Notify user only on meaningful action changes above confidence threshold

Usage:
  python run.py              # live mode
  python run.py --paper      # paper trading mode
  python run.py --debug      # verbose debug output
  python run.py --status     # show current positions and exit
  python run.py --weights    # show agent weight summary and exit
=============================================================================
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ─── Add scanner directory to path ───────────────────────────────────────────
# Assumes options_scanner.py is in the parent directory or same directory.
# Adjust SCANNER_DIR if your layout differs.
SCANNER_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCANNER_DIR))

# ─── Local imports ────────────────────────────────────────────────────────────
import config as cfg
import database as db
from database import _dumps as _json_dumps  # numpy-safe json encoder
from rl_agent import DecisionAgent
from position_tracker import PositionTracker
from notifier import (
    Alert, send, notify_entry, notify_exit,
    notify_stop_hit, notify_info
)

# ─── Scanner imports ──────────────────────────────────────────────────────────
try:
    from options_scanner import (
        analyze_ticker,
        determine_market_regime,
        WATCHLIST,
    )
    SCANNER_AVAILABLE = True
except ImportError as e:
    SCANNER_AVAILABLE = False
    print(f"WARNING: Could not import options_scanner: {e}")
    print("Ensure options_scanner.py is in the parent directory.")

# ─── Logging setup ────────────────────────────────────────────────────────────
Path(cfg.LOG_DIR).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(Path(cfg.LOG_DIR) / "run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run")


# =============================================================================
#  MARKET SNAPSHOT BUILDER
# =============================================================================

def build_market_snapshot(scanner_result: Dict, position: Optional[Dict] = None,
                           current_option_price: float = None) -> Dict:
    """
    Build a standardized market snapshot dict for the agent and DB.
    Pulls from scanner result + optional live position context.
    """
    tech    = scanner_result.get("tech", {})
    vol     = scanner_result.get("vol", {})
    sector  = scanner_result.get("sector", {})
    regime  = scanner_result.get("regime_data", {})
    flow    = scanner_result.get("flow", [])
    pricing = scanner_result.get("pricing", {})
    trade   = scanner_result.get("trade", {})
    main    = trade.get("main_leg", {})

    snapshot = {
        "timestamp":       datetime.now().isoformat(),
        "ticker":          scanner_result.get("ticker"),
        "spot":            scanner_result.get("spot"),
        "rsi":             tech.get("rsi"),
        "above_vwap":      tech.get("above_vwap"),
        "ema_cross":       tech.get("ema_cross"),
        "ivr_current":     vol.get("ivr"),
        "iv_current":      vol.get("current_iv"),
        "spy_change_pct":  sector.get("spy_change_pct"),
        "qqq_change_pct":  sector.get("qqq_change_pct"),
        "regime":          regime.get("regime") if isinstance(regime, dict) else None,
        "flow_score":      flow[0].get("score") if flow else 0,
        "theta_today":     abs(main.get("theta", 0) or 0),
        "option_mid":      current_option_price or main.get("mid"),
    }

    # Position-relative fields
    if position:
        entry_time = position.get("entry_time", datetime.now().isoformat())
        try:
            entry_dt   = datetime.fromisoformat(entry_time)
            days_held  = (datetime.now() - entry_dt).total_seconds() / 86400
        except Exception:
            days_held  = 0

        exp = position.get("expiration", "")
        dte_rem = None
        try:
            exp_dt  = datetime.strptime(exp, "%Y-%m-%d")
            dte_rem = (exp_dt - datetime.now()).days
        except Exception:
            pass

        entry_cost = position.get("entry_cost", 1)
        opt_mid    = current_option_price or main.get("mid") or position.get("entry_price")
        contracts  = position.get("contracts", 1)
        cur_val    = (opt_mid or 0) * 100 * contracts
        pnl        = cur_val - entry_cost
        r          = pnl / entry_cost if entry_cost > 0 else 0

        snapshot.update({
            "days_since_entry":  days_held,
            "dte_remaining":     dte_rem,
            "option_mid":        opt_mid,
            "unrealized_pnl":    pnl,
            "unrealized_r":      r,
        })

    return snapshot


# =============================================================================
#  PRIOR ACTION STATE
# =============================================================================

class ActionStateTracker:
    """
    Tracks the last recommended action per position/candidate
    to detect changes and avoid duplicate alerts.
    """
    def __init__(self):
        self._last_action: Dict[str, str] = {}
        self._last_confidence: Dict[str, float] = {}
        # Restore from DB
        saved = db.get_state("action_state") or {}
        self._last_action = saved.get("actions", {})
        self._last_confidence = saved.get("confidences", {})

    def has_changed(self, key: str, new_action: str) -> bool:
        return self._last_action.get(key) != new_action

    def update(self, key: str, action: str, confidence: float):
        self._last_action[key]     = action
        self._last_confidence[key] = confidence
        db.set_state("action_state", {
            "actions":     self._last_action,
            "confidences": self._last_confidence
        })

    def get_last(self, key: str) -> Optional[str]:
        return self._last_action.get(key)


# =============================================================================
#  COPY-PASTE COMMAND GENERATOR
# =============================================================================

def build_track_command(scanner_result: Dict) -> str:
    """
    Build the exact copy-paste command the user needs to record
    this trade in the position tracker after executing in ThinkorSwim.
    Printed automatically with every ENTER alert.
    """
    trade     = scanner_result.get("trade", {})
    pricing   = scanner_result.get("pricing", {})
    main      = trade.get("main_leg", {})
    ticker    = scanner_result.get("ticker", "?")
    strategy  = trade.get("strategy", "?")
    strike    = main.get("strike", 0)
    exp       = main.get("exp", trade.get("exp", "?"))
    contracts = pricing.get("contracts", 1)

    q = '"'
    lines = [
        "python3 -c " + q,
        "import sys; sys.path.insert(0, 'rl_system')",
        "from position_tracker import PositionTracker",
        "t = PositionTracker()",
        "t.manual_override_open(",
        "    ticker='" + str(ticker) + "',",
        "    strategy='" + str(strategy) + "',",
        "    strike=" + str(strike) + ",",
        "    expiration='" + str(exp) + "',",
        "    entry_price=FILL_PRICE,  # <-- replace with your actual fill",
        "    contracts=" + str(contracts),
        ")",
        q
    ]
    return "\n".join(lines)


def build_close_command(position_id: int, ticker: str) -> str:
    """
    Build the copy-paste command to close a position after
    executing the exit in ThinkorSwim.
    Printed automatically with every EXIT alert.
    """
    q = '"'
    lines = [
        "python3 -c " + q,
        "import sys; sys.path.insert(0, 'rl_system')",
        "from position_tracker import PositionTracker",
        "t = PositionTracker()",
        "t.manual_override_close(",
        "    position_id=" + str(position_id) + ",",
        "    current_price=FILL_PRICE,  # <-- replace with your actual exit fill",
        "    reason='MANUAL'",
        ")",
        q
    ]
    return "\n".join(lines)


# =============================================================================
#  TIMED INPUT HELPER
# =============================================================================

def timed_input(prompt: str, timeout: int = 60, default: str = "n") -> str:
    """
    Ask the user a question with a timeout.
    Uses signal.alarm on Mac/Linux for reliable stdin reading.
    Falls back to direct input() on Windows (no timeout on Windows).
    """
    import sys
    import platform

    sys.stdout.write(prompt)
    sys.stdout.flush()

    # ── Mac/Linux: use signal.alarm for reliable timeout
    if platform.system() != "Windows":
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            val = input().strip().lower()
            signal.alarm(0)  # cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
            return val if val else default
        except TimeoutError:
            print(f"\n  (No response — defaulting to '{default}')")
            signal.signal(signal.SIGALRM, old_handler)
            return default
        except Exception:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return default

    # ── Windows: no signal.alarm available — just use input() with no timeout
    # On Windows the loop will pause until user responds
    else:
        try:
            val = input().strip().lower()
            return val if val else default
        except Exception:
            return default


def ask_track_position(scanner_result: Dict, tracker: PositionTracker,
                        confidence: float) -> Optional[int]:
    """
    Ask user if they want to track this position.
    If yes, ask for fill price and open the position.
    Returns position_id if opened, None otherwise.
    Times out after 30 seconds and defaults to n.
    """
    ticker  = scanner_result.get("ticker", "?")
    pricing = scanner_result.get("pricing", {})
    suggested_entry = pricing.get("entry", 0)

    sep = "=" * 64
    print("\n" + sep)
    print(f"  Did you execute this trade in ThinkorSwim?")
    response = timed_input(
        f"  Track {ticker} position? (y/n, 30s timeout): ",
        timeout=30, default="n"
    )

    if response != "y":
        print(f"  Skipped — not tracking {ticker}")
        print(sep + "\n")
        return None

    # Ask for fill price
    fill_str = timed_input(
        f"  What did you pay? (suggested ${suggested_entry}, press enter to use): ",
        timeout=30, default=str(suggested_entry)
    )

    try:
        fill_price = float(fill_str) if fill_str else suggested_entry
    except ValueError:
        fill_price = suggested_entry
        print(f"  Could not parse price — using suggested ${suggested_entry}")

    # Open the position
    position_id = tracker.open_position(scanner_result)
    if position_id:
        # Update with actual fill price
        entry_cost = fill_price * 100 * pricing.get("contracts", 1)
        stop_price  = round(fill_price * (1 - cfg.STOP_LOSS_PCT), 2)
        target_price = round(fill_price * (1 + cfg.PROFIT_TARGET_PCT), 2)
        with db.get_connection() as conn:
            conn.execute(
                """UPDATE positions SET
                   entry_price = ?, entry_cost = ?,
                   stop_price = ?, target_price = ?
                   WHERE id = ?""",
                (fill_price, entry_cost, stop_price, target_price, position_id)
            )
        print(f"  ✓ {ticker} tracked — entry=${fill_price} "
              f"stop=${stop_price} target=${target_price}")
        db.log_journal_event(
            "POSITION_OPENED", ticker=ticker, position_id=position_id,
            action="ENTER", confidence=confidence,
            reason_summary=f"User confirmed entry: {ticker} @ ${fill_price}",
            details={"fill_price": fill_price, "suggested": suggested_entry}
        )
    else:
        print(f"  Could not open position — check logs")

    print(sep + "\n")
    return position_id


def ask_close_position(position_id: int, position: Dict, tracker: PositionTracker,
                        current_price: float, confidence: float,
                        reason: str) -> bool:
    """
    Ask user if they want to close this position.
    If yes, ask for exit fill price and close it.
    Returns True if closed, False if kept open.
    Times out after 30 seconds and defaults to n.
    """
    ticker = position.get("ticker", "?")
    pnl    = tracker.unrealized_pnl(position_id, current_price)
    r      = tracker.unrealized_r(position_id, current_price)
    sign   = "+" if pnl >= 0 else ""

    sep = "=" * 64
    print("\n" + sep)
    print(f"  Current P&L: {sign}${pnl:.2f} ({r:+.2f}R)")
    response = timed_input(
        f"  Close {ticker} position? (y/n, 30s timeout): ",
        timeout=30, default="n"
    )

    if response != "y":
        print(f"  Keeping {ticker} open — continuing to monitor")
        print(sep + "\n")
        return False

    # Ask for exit fill price
    fill_str = timed_input(
        f"  What did you close at? (suggested ${current_price:.2f}, press enter to use): ",
        timeout=30, default=str(round(current_price, 2))
    )

    try:
        exit_price = float(fill_str) if fill_str else current_price
    except ValueError:
        exit_price = current_price
        print(f"  Could not parse price — using ${current_price:.2f}")

    result = tracker.close_position(position_id, exit_price, reason)
    realized_pnl = result.get("realized_pnl", 0)
    realized_r   = result.get("realized_r", 0)
    sign = "+" if realized_pnl >= 0 else ""

    print(f"  ✓ {ticker} closed @ ${exit_price} — "
          f"P&L: {sign}${realized_pnl:.2f} ({realized_r:+.2f}R)")
    print(sep + "\n")
    return True


# =============================================================================
#  LIVE PRICE FETCHER FOR OPEN POSITIONS
# =============================================================================

# Simple price cache — stores (price, timestamp) per position_id
# Avoids hammering Tradier API every tick for every open position
_price_cache: Dict[int, tuple] = {}
_PRICE_CACHE_TTL = 55  # seconds — refresh just before each tick

def get_current_option_price(position: Dict) -> Optional[float]:
    """
    Fetch the actual current mid price for a specific open contract
    using Tradier or yfinance.

    Caches results for ~55 seconds to avoid rate limiting Tradier.
    Falls back to entry_price if live data is unavailable.
    """
    import os
    import requests as req

    position_id = position.get("id")

    # ── Check cache first
    if position_id and position_id in _price_cache:
        cached_price, cached_time = _price_cache[position_id]
        age = (datetime.now() - cached_time).total_seconds()
        if age < _PRICE_CACHE_TTL:
            return cached_price

    ticker     = position.get("ticker", "")
    strike     = position.get("strike")
    expiration = position.get("expiration")
    opt_type   = position.get("option_type", "call")
    entry_price = position.get("entry_price", 0)

    if not ticker or not strike or not expiration:
        return entry_price

    def _cache_and_return(price: float) -> float:
        if position_id:
            _price_cache[position_id] = (price, datetime.now())
        return price

    # ── Try Tradier first
    api_key = os.environ.get("TRADIER_API_KEY", "")
    if api_key:
        try:
            base = "https://api.tradier.com/v1"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
            r = req.get(
                f"{base}/markets/options/chains",
                headers=headers,
                params={
                    "symbol":     ticker,
                    "expiration": expiration,
                    "greeks":     "false"
                },
                timeout=8
            )
            if r.status_code == 200:
                options = r.json().get("options", {}).get("option", [])
                for c in options:
                    if (abs(float(c.get("strike", 0)) - float(strike)) < 0.01 and
                            c.get("option_type", "").lower() == opt_type.lower()):
                        bid = float(c.get("bid", 0) or 0)
                        ask = float(c.get("ask", 0) or 0)
                        if bid > 0 and ask > 0:
                            return _cache_and_return(round((bid + ask) / 2, 2))
                        last = float(c.get("last", 0) or 0)
                        if last > 0:
                            return _cache_and_return(round(last, 2))
        except Exception as e:
            logger.debug(f"Tradier price fetch failed for {ticker}: {e}")

    # ── Fallback: yfinance option chain
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiration)
        df = chain.calls if opt_type.lower() == "call" else chain.puts
        row = df[abs(df["strike"] - float(strike)) < 0.01]
        if not row.empty:
            bid = float(row.iloc[0]["bid"] or 0)
            ask = float(row.iloc[0]["ask"] or 0)
            if bid > 0 and ask > 0:
                return _cache_and_return(round((bid + ask) / 2, 2))
    except Exception as e:
        logger.debug(f"yfinance price fetch failed for {ticker}: {e}")

    # ── Last resort: return entry price so nothing breaks
    logger.debug(f"Could not fetch live price for {ticker} {strike} {expiration} — using entry price")
    return entry_price


# =============================================================================
#  CORE LOOP FUNCTIONS
# =============================================================================

def run_scanner(paper_mode: bool, regime: Dict) -> List[Dict]:
    """Run the full options scanner and return valid results."""
    if not SCANNER_AVAILABLE:
        logger.warning("Scanner not available — skipping scan")
        return []

    logger.info(f"Running options scanner on {len(WATCHLIST)} tickers...")
    results = []
    for ticker in WATCHLIST:
        try:
            result = analyze_ticker(ticker, regime, paper_mode=paper_mode, verbose=False)
            results.append(result)
            time.sleep(0.3)   # rate limiting
        except Exception as e:
            logger.warning(f"Scanner error on {ticker}: {e}")

    valid = [
        r for r in results
        if not r.get("error")
        and r.get("confluence", {}).get("score", 0) >= cfg.MIN_CONFLUENCE_SCORE_TO_ENTER
        and r.get("trade", {}).get("main_leg", {}).get("strike")
        and (r.get("pricing", {}).get("entry") or 0) >= 0.50
    ]

    logger.info(f"Scanner complete: {len(results)} scanned, {len(valid)} meet threshold")
    return sorted(valid, key=lambda r: r.get("confluence", {}).get("score", 0), reverse=True)


def evaluate_open_positions(
    tracker: PositionTracker,
    agent: DecisionAgent,
    action_state: ActionStateTracker,
    scanner_results: List[Dict],
    regime: Dict
) -> List[Dict]:
    """
    Evaluate each open position. Apply hard rules first, then agent scoring.
    Returns list of action dicts taken this tick.
    """
    actions_taken = []

    # Build a lookup from ticker → latest scanner result
    scanner_by_ticker = {
        r.get("ticker", "").upper(): r for r in scanner_results
    }

    for position_id, position in list(tracker.open_positions.items()):
        ticker = position["ticker"].upper()

        # Get latest scanner data for this ticker if available,
        # otherwise use a minimal stub
        scanner_result = scanner_by_ticker.get(ticker, {
            "ticker": ticker, "tech": {}, "vol": {}, "sector": {},
            "flow": [], "pricing": {}, "trade": {"main_leg": {}},
            "confluence": {}, "regime_data": regime
        })
        scanner_result["regime_data"] = regime

        # Get ACTUAL current price for this specific contract
        # Do NOT use the scanner's main_leg.mid — that's a different strike/expiry
        # Always fetch the live mid for the exact contract we hold
        current_option_price = get_current_option_price(position)

        # Build snapshot
        snapshot = build_market_snapshot(scanner_result, position, current_option_price)

        # ── HARD RULES FIRST — non-negotiable
        must_exit, hard_reason = tracker.check_hard_exit_rules(position, current_option_price)

        if must_exit:
            # ── Notify user FIRST before closing
            if "STOP" in hard_reason.upper():
                notify_stop_hit(
                    ticker=ticker,
                    loss=tracker.unrealized_pnl(position_id, current_option_price),
                    position_id=position_id
                )
            else:
                notify_exit(
                    ticker=ticker,
                    confidence=1.0,
                    reasons=[f"Hard rule: {hard_reason}"],
                    unrealized_pnl=tracker.unrealized_pnl(position_id, current_option_price),
                    force=True
                )

            # ── Ask for actual fill price BEFORE closing position
            sep = "=" * 64
            print(f"\n{sep}")
            print(f"  !! HARD RULE TRIGGERED — CLOSE {ticker} IN THINKORSWIM NOW !!")
            print(f"  Reason: {hard_reason}")
            print(sep)
            fill_str = timed_input(
                f"  What did you close {ticker} at? (suggested ${current_option_price:.2f}, press enter): ",
                timeout=60, default=str(round(current_option_price, 2))
            )
            try:
                confirmed_exit_price = float(fill_str) if fill_str else current_option_price
            except ValueError:
                confirmed_exit_price = current_option_price
            print(sep + "\n")

            # ── Now close with the ACTUAL fill price
            result = tracker.close_position(position_id, confirmed_exit_price, hard_reason)
            realized_r = result.get("realized_r", 0)

            # Update agent
            entry_snapshot = json.loads(
                position.get("raw_scanner_data") or "{}"
            )
            agent.update_on_close(
                position=position,
                exit_reason=hard_reason,
                realized_r=realized_r,
                entry_market_snapshot=build_market_snapshot(entry_snapshot),
                exit_market_snapshot=snapshot,
                ticks_held=_ticks_held(position),
                rolling_drawdown=_rolling_drawdown()
            )

            action_state.update(str(position_id), "EXIT", 1.0)
            actions_taken.append({
                "type": "HARD_EXIT", "ticker": ticker,
                "reason": hard_reason, "position_id": position_id
            })
            continue   # position is closed — skip agent scoring

        # ── AGENT SCORING — can recommend EXIT or HOLD
        ticks = _ticks_held(position)

        # Grace period: skip agent exit scoring for first 5 minutes after entry
        # Without live Tradier prices the snapshot data is unreliable immediately
        # after opening and the agent may recommend EXIT on bad data
        entry_time = position.get("entry_time", "")
        in_grace_period = False
        if entry_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time)
                secs_held = (datetime.now() - entry_dt).total_seconds()
                if secs_held < 300:  # 5 minute grace period
                    in_grace_period = True
            except Exception:
                pass

        if in_grace_period:
            action     = "HOLD"
            confidence = 0.0
            reasons    = ["Grace period — position too new to evaluate (< 5 min)"]
        else:
            action, confidence, reasons = agent.score_exit(position, snapshot, ticks)

        # Persist tick snapshot to DB
        db.insert_tick_snapshot({
            "position_id":    position_id,
            "ticker":         ticker,
            "timestamp":      snapshot["timestamp"],
            "current_price":  snapshot.get("spot"),
            "option_mid":     current_option_price,
            "unrealized_pnl": snapshot.get("unrealized_pnl"),
            "unrealized_r":   snapshot.get("unrealized_r"),
            "dte_remaining":  snapshot.get("dte_remaining"),
            "theta_today":    snapshot.get("theta_today"),
            "iv_current":     snapshot.get("iv_current"),
            "ivr_current":    snapshot.get("ivr_current"),
            "spy_change_pct": snapshot.get("spy_change_pct"),
            "rsi":            snapshot.get("rsi"),
            "above_vwap":     1 if snapshot.get("above_vwap") else 0,
            "regime":         snapshot.get("regime"),
            "flow_score":     snapshot.get("flow_score"),
            "feature_vector": _json_dumps([]),   # populated in agent internally
            "agent_action":   action,
            "agent_confidence": confidence,
            "agent_reasons":  _json_dumps(reasons[:5])
        })

        # Alert if action changed and confidence is high enough
        key = str(position_id)
        changed = action_state.has_changed(key, action)

        should_notify = (
            changed and
            action == "EXIT" and
            confidence >= cfg.NOTIFY_CONFIDENCE_THRESHOLD
        )

        if should_notify:
            notify_exit(
                ticker=ticker,
                confidence=confidence,
                reasons=reasons[:5],
                unrealized_pnl=snapshot.get("unrealized_pnl"),
                exit_price=current_option_price
            )

            # ── Ask user if they want to close this position (30s timeout → n)
            closed = ask_close_position(
                position_id=position_id,
                position=position,
                tracker=tracker,
                current_price=current_option_price,
                confidence=confidence,
                reason="AGENT_EXIT"
            )

            # If user closed it, update agent and skip rest of loop for this position
            if closed:
                result = {
                    "realized_pnl": tracker.unrealized_pnl(position_id, current_option_price)
                    if position_id in tracker.open_positions else
                    snapshot.get("unrealized_pnl", 0),
                    "realized_r": snapshot.get("unrealized_r", 0)
                }
                entry_snap = json.loads(position.get("raw_scanner_data") or "{}")
                agent.update_on_close(
                    position=position,
                    exit_reason="AGENT_EXIT",
                    realized_r=snapshot.get("unrealized_r", 0),
                    entry_market_snapshot=build_market_snapshot(entry_snap),
                    exit_market_snapshot=snapshot,
                    ticks_held=_ticks_held(position),
                    rolling_drawdown=_rolling_drawdown()
                )
                action_state.update(str(position_id), "EXIT", confidence)

            db.insert_recommendation({
                "timestamp":      snapshot["timestamp"],
                "ticker":         ticker,
                "strategy":       position.get("strategy"),
                "direction":      position.get("direction"),
                "strike":         position.get("strike"),
                "expiration":     position.get("expiration"),
                "action":         action,
                "confidence":     confidence,
                "reasons":        _json_dumps(reasons[:5]),
                "market_snapshot": _json_dumps(snapshot),
                "notified_user":  1,
                "position_id":    position_id
            })

        action_state.update(key, action, confidence)

        if DEBUG_MODE or changed:
            pnl_str = f"${snapshot.get('unrealized_pnl', 0):+.2f}"
            logger.info(
                f"  {ticker} #{position_id}: {action} "
                f"conf={confidence:.2f} pnl={pnl_str}"
            )

    return actions_taken


def evaluate_new_candidates(
    tracker: PositionTracker,
    agent: DecisionAgent,
    action_state: ActionStateTracker,
    scanner_results: List[Dict],
    regime: Dict
) -> List[Dict]:
    """
    Evaluate scanner recommendations for potential new entries.
    Hard rules checked first, then agent scores ENTER vs WAIT.
    """
    actions_taken = []

    for scanner_result in scanner_results[:5]:   # top 5 candidates only
        ticker = scanner_result.get("ticker", "?").upper()
        scanner_result["regime_data"] = regime

        # Skip if already in a position on this ticker
        if tracker.is_open(ticker):
            # Don't reset action state here — let it expire naturally
            # Resetting to HOLD/0.0 would prevent ENTER from firing
            # again after the position closes
            continue

        # Skip if last action for this ticker was already ENTER
        # prevents re-alerting on same signal before user has had time to act
        last_action = action_state.get_last(f"entry_{ticker}")
        if last_action == "ENTER":
            continue

        # Skip if on cooldown
        if db.is_on_cooldown(ticker):
            cd = db.get_cooldown(ticker)
            if DEBUG_MODE:
                logger.debug(f"  {ticker} on cooldown until {cd.get('cooldown_until', '?')[:16]}")
            continue

        # Skip if max positions reached
        if tracker.open_count >= cfg.MAX_CONCURRENT_POSITIONS:
            logger.info(f"Max positions ({cfg.MAX_CONCURRENT_POSITIONS}) reached — skipping {ticker}")
            break

        snapshot = build_market_snapshot(scanner_result)

        # Agent scores the entry
        action, confidence, reasons = agent.score_entry(scanner_result, snapshot)

        key = f"entry_{ticker}"
        changed = action_state.has_changed(key, action)

        should_notify = (
            changed and
            action == "ENTER" and
            confidence >= cfg.NOTIFY_CONFIDENCE_THRESHOLD
        )

        if should_notify:
            pricing  = scanner_result.get("pricing", {})
            trade    = scanner_result.get("trade", {})
            main_leg = trade.get("main_leg", {})
            short_leg = trade.get("short_leg", {})
            strategy  = trade.get("strategy", "")
            exp       = main_leg.get("exp") or trade.get("exp")
            opt_label = main_leg.get("option_type", "put").upper()

            # Build a clear trade description showing both legs for spreads
            if "SPREAD" in strategy and short_leg:
                long_strike  = main_leg.get("strike")
                short_strike = short_leg.get("strike")
                long_mid     = main_leg.get("mid", 0)
                short_mid    = short_leg.get("mid", 0)
                net_debit    = pricing.get("entry", round(long_mid - short_mid, 2))
                spread_width = abs((long_strike or 0) - (short_strike or 0))
                max_profit   = round(spread_width - net_debit, 2) if spread_width else "?"
                be = (
                    round(long_strike - net_debit, 2) if "BEAR" in strategy and long_strike
                    else round(long_strike + net_debit, 2) if long_strike else "?"
                )
                trade_summary = (
                    f"Buy ${long_strike} {opt_label} / Sell ${short_strike} {opt_label}  exp {exp}\n"
                    f"         Net debit: ${net_debit}  |  Max profit: ${max_profit}  |  Break-even: ${be}"
                )
            else:
                trade_summary = f"${main_leg.get('strike')} {opt_label}  exp {exp}"

            notify_entry(
                ticker=ticker,
                confidence=confidence,
                reasons=reasons[:5],
                strategy=strategy,
                strike=main_leg.get("strike"),
                expiration=exp,
                entry=pricing.get("entry"),
                stop=pricing.get("stop"),
                target=pricing.get("target"),
                contracts=pricing.get("contracts"),
                trade_summary=trade_summary
            )

            # ── Ask user if they want to track this position (30s timeout → n)
            position_id = ask_track_position(scanner_result, tracker, confidence)

            # If user tracked it, immediately flip action_state to HOLD
            # so the ENTER alert doesn't fire again on the next tick
            if position_id is not None:
                action_state.update(key, "HOLD", confidence)
                logger.info(f"  {ticker} now tracked — action state set to HOLD")

            rec_id = db.insert_recommendation({
                "timestamp":       snapshot["timestamp"],
                "ticker":          ticker,
                "strategy":        trade.get("strategy"),
                "direction":       trade.get("direction"),
                "strike":          trade.get("main_leg", {}).get("strike"),
                "expiration":      trade.get("exp"),
                "action":          action,
                "confidence":      confidence,
                "reasons":         _json_dumps(reasons[:5]),
                "market_snapshot": _json_dumps(snapshot),
                "notified_user":   1,
                "position_id":     position_id
            })

            db.log_journal_event(
                "ENTRY_REC",
                ticker=ticker,
                action=action,
                confidence=confidence,
                reason_summary=f"ENTER {ticker} conf={confidence:.2f}",
                details={
                    "reasons":    reasons[:5],
                    "entry":      pricing.get("entry"),
                    "stop":       pricing.get("stop"),
                    "target":     pricing.get("target"),
                    "strategy":   trade.get("strategy"),
                    "confluence": scanner_result.get("confluence", {}).get("score"),
                    "rec_id":     rec_id,
                    "tracked":    position_id is not None
                }
            )

        action_state.update(key, action, confidence)

        if DEBUG_MODE or (changed and action == "ENTER"):
            logger.info(
                f"  Candidate {ticker}: {action} conf={confidence:.2f} "
                f"(confluence {scanner_result.get('confluence', {}).get('score', 0)}pts)"
            )

        actions_taken.append({
            "type": action, "ticker": ticker,
            "confidence": confidence
        })

    return actions_taken


# =============================================================================
#  HELPERS
# =============================================================================

def _ticks_held(position: Dict) -> int:
    """Estimate ticks held based on entry time and loop interval."""
    try:
        entry_dt = datetime.fromisoformat(position["entry_time"])
        elapsed  = (datetime.now() - entry_dt).total_seconds()
        return int(elapsed / cfg.LOOP_INTERVAL_SECONDS)
    except Exception:
        return 0


def _rolling_drawdown() -> float:
    """
    Compute recent rolling drawdown from closed positions.
    Returns a negative value when in drawdown, 0 when flat or profitable.
    Only negative P&L contributes — winners don't offset drawdown here.
    """
    recent = db.get_closed_positions(limit=10)
    if not recent:
        return 0.0
    # Only count losses — drawdown is about downside, not net P&L
    losses = sum(p.get("realized_pnl", 0) for p in recent if p.get("realized_pnl", 0) < 0)
    return losses / cfg.ACCOUNT_SIZE


def print_status(tracker: PositionTracker, agent: DecisionAgent):
    """Print current system status to terminal."""
    print("\n" + "═" * 60)
    print("  OPTIONS SCANNER — SYSTEM STATUS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    print(f"\n  Open positions ({tracker.open_count}/{cfg.MAX_CONCURRENT_POSITIONS}):")
    tracker.print_summary()

    perf = db.get_performance_summary()
    if perf.get("total_trades"):
        wr = perf.get("winners", 0) / max(perf.get("total_trades", 1), 1) * 100
        print(f"\n  Performance ({perf.get('total_trades')} closed trades):")
        print(f"    Win rate:   {wr:.1f}%")
        print(f"    Avg R:      {perf.get('avg_r', 0):.3f}R")
        print(f"    Total P&L:  ${perf.get('total_pnl', 0):+,.2f}")
        print(f"    Best:       {perf.get('best_r', 0):.2f}R")
        print(f"    Worst:      {perf.get('worst_r', 0):.2f}R")

    print(f"\n  Agent:")
    print(f"    Enter model updates: {agent.enter_model.n_updates}")
    print(f"    Exit model updates:  {agent.exit_model.n_updates}")
    print(f"    Enter mean reward:   {agent.enter_model.mean_reward:.3f}R")
    print(f"    Exit mean reward:    {agent.exit_model.mean_reward:.3f}R")
    print()


# =============================================================================
#  GRACEFUL SHUTDOWN
# =============================================================================

_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received — finishing current tick...")
    _running = False


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

def main():
    global DEBUG_MODE

    parser = argparse.ArgumentParser(description="Options Scanner RL Loop")
    parser.add_argument("--paper",   action="store_true", help="Paper trading mode")
    parser.add_argument("--debug",   action="store_true", help="Debug/verbose mode")
    parser.add_argument("--status",  action="store_true", help="Show status and exit")
    parser.add_argument("--weights", action="store_true", help="Show agent weights and exit")
    args = parser.parse_args()

    if args.debug:
        cfg.DEBUG_MODE = True
        logging.getLogger().setLevel(logging.DEBUG)

    DEBUG_MODE = cfg.DEBUG_MODE

    # ── Initialize
    db.initialize_database()
    tracker      = PositionTracker()
    agent        = DecisionAgent()
    action_state = ActionStateTracker()

    # ── One-shot commands
    if args.status:
        print_status(tracker, agent)
        return

    if args.weights:
        agent.print_weight_summary()
        return

    # ── Startup log
    db.log_journal_event(
        "SYSTEM_START",
        reason_summary="RL loop started",
        details={
            "paper_mode":          args.paper,
            "debug_mode":          args.debug,
            "loop_interval":       cfg.LOOP_INTERVAL_SECONDS,
            "scanner_interval":    cfg.SCANNER_RUN_INTERVAL,
            "max_positions":       cfg.MAX_CONCURRENT_POSITIONS,
            "open_positions":      tracker.open_count,
            "enter_model_updates": agent.enter_model.n_updates,
            "exit_model_updates":  agent.exit_model.n_updates,
        }
    )

    notify_info(
        f"RL loop started — "
        f"{'PAPER' if args.paper else 'LIVE'} mode, "
        f"{tracker.open_count} positions restored"
    )

    print_status(tracker, agent)

    # ── Main loop state
    last_scanner_run   = 0.0
    last_scanner_results: List[Dict] = []
    regime: Dict = {}
    tick = 0

    logger.info(
        f"Starting 60-second loop "
        f"({'PAPER' if args.paper else 'LIVE'} mode)"
    )

    while _running:
        tick_start = time.time()
        tick += 1
        now = datetime.now()

        try:
            # ── Scanner refresh
            time_since_scan = tick_start - last_scanner_run
            if time_since_scan >= cfg.SCANNER_RUN_INTERVAL or not last_scanner_results:
                if SCANNER_AVAILABLE:
                    try:
                        regime = determine_market_regime()
                    except Exception as e:
                        logger.warning(f"Regime detection failed: {e}")
                        regime = {"regime": "UNKNOWN"}

                    last_scanner_results = run_scanner(args.paper, regime)
                    last_scanner_run = tick_start

                    logger.info(
                        f"Tick {tick}: Scanner refreshed — "
                        f"regime={regime.get('regime')} "
                        f"VIX={regime.get('vix')} "
                        f"{len(last_scanner_results)} valid candidates"
                    )
                else:
                    logger.warning("Scanner unavailable — using cached results")

            # ── Evaluate open positions
            if tracker.open_count > 0:
                evaluate_open_positions(
                    tracker, agent, action_state, last_scanner_results, regime
                )

            # ── Evaluate new entry candidates
            evaluate_new_candidates(
                tracker, agent, action_state, last_scanner_results, regime
            )

            # ── Periodic weight save
            if tick % cfg.AGENT_SAVE_INTERVAL_TICKS == 0:
                agent.save_weights()

            # ── Periodic status print (every 10 minutes)
            if tick % (600 // cfg.LOOP_INTERVAL_SECONDS) == 0:
                print_status(tracker, agent)

        except Exception as e:
            logger.error(f"Loop error on tick {tick}: {e}")
            if DEBUG_MODE:
                traceback.print_exc()

        # ── Sleep for remainder of interval
        elapsed = time.time() - tick_start
        sleep_for = max(0, cfg.LOOP_INTERVAL_SECONDS - elapsed)
        if DEBUG_MODE:
            logger.debug(
                f"Tick {tick} complete in {elapsed:.1f}s — "
                f"sleeping {sleep_for:.1f}s"
            )
        time.sleep(sleep_for)

    # ── Shutdown
    agent.save_weights()
    db.log_journal_event(
        "SYSTEM_STOP",
        reason_summary="RL loop stopped",
        details={"tick": tick, "open_positions": tracker.open_count}
    )
    notify_info("RL loop stopped — weights saved")
    logger.info(f"Loop stopped after {tick} ticks")


if __name__ == "__main__":
    main()
