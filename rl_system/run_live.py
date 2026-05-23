"""
=============================================================================
run_live.py — Live Trading Mode (mid-price limits, separate Alpaca account)
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
cfg.DB_PATH = "./scanner_data_live.db"  # Separate DB for mid-price live simulation
import database as db
# Broker is imported dynamically based on --broker flag
# Default to alpaca broker; overridden after arg parsing
try:
    import broker
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False
# zoneinfo imported inside is_market_hours_for_entry() for Python 3.9+ compatibility
from database import _dumps as _json_dumps  # numpy-safe json encoder
from database import (
    save_pending_orders, load_pending_orders,
    save_pending_exit_orders, load_pending_exit_orders,
)
from rl_agent import DecisionAgent

# ─── OI cache for change detection (Tier 2 feature) ─────────────────────────
_oi_cache: Dict[str, int] = {}   # "TICKER:strike:exp:type" → last known OI
# Earnings cache to avoid hitting API every tick
_earnings_cache: Dict[str, Optional[str]] = {}   # ticker → next earnings date ISO or None
_earnings_cache_time: Dict[str, float] = {}       # ticker → timestamp of last fetch
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
        get_live_option_quote,
    )
    SCANNER_AVAILABLE = True
except ImportError as e:
    SCANNER_AVAILABLE = False
    get_live_option_quote = None
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
#  TIME-OF-DAY FILTER
# =============================================================================

def is_market_hours_for_entry() -> bool:
    """
    Returns True if current ET time is within the allowed entry window.
    Hard exits (stop/target/DTE) always fire regardless — this only
    suppresses new ENTER recommendations.

    Window: 10:00am ET to 3:30pm ET (configurable in config.py)
    Outside this window options spreads have:
      - Wider bid/ask (costs more to fill)
      - Lower liquidity (harder to exit)
      - Erratic prints (first 30min after open)
    """
    if not cfg.ENFORCE_MARKET_HOURS:
        return True

    try:
        try:
            from zoneinfo import ZoneInfo          # Python 3.9+
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # pip install backports.zoneinfo
        et = ZoneInfo("America/New_York")
        now_et = datetime.now(et)
        # Monday=0 ... Friday=4, Saturday=5, Sunday=6
        if now_et.weekday() >= 5:
            return False   # weekend — market closed
        open_time  = now_et.replace(
            hour=cfg.MARKET_OPEN_HOUR, minute=cfg.MARKET_OPEN_MINUTE,
            second=0, microsecond=0
        )
        close_time = now_et.replace(
            hour=cfg.MARKET_CLOSE_HOUR, minute=cfg.MARKET_CLOSE_MINUTE,
            second=0, microsecond=0
        )
        return open_time <= now_et <= close_time
    except Exception as e:
        logger.warning(f"Market hours check failed: {e} — defaulting to closed")
        return False


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

    States expire after ACTION_STATE_EXPIRY_HOURS (default 20h) so that
    a signal seen yesterday doesn't suppress the same signal today.
    """
    def __init__(self):
        self._last_action: Dict[str, str] = {}
        self._last_confidence: Dict[str, float] = {}
        self._last_updated: Dict[str, str] = {}   # key → ISO timestamp
        # Restore from DB
        saved = db.get_state("action_state") or {}
        self._last_action    = saved.get("actions", {})
        self._last_confidence = saved.get("confidences", {})
        self._last_updated   = saved.get("updated_at", {})
        # Expire stale states on startup
        self._expire_stale()

    def _expire_stale(self):
        """Remove action states older than ACTION_STATE_EXPIRY_HOURS."""
        cutoff = datetime.now() - timedelta(hours=cfg.ACTION_STATE_EXPIRY_HOURS)
        stale = []
        for key, ts_str in self._last_updated.items():
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts < cutoff:
                    stale.append(key)
            except Exception:
                stale.append(key)   # unparseable timestamp — expire it
        for key in stale:
            self._last_action.pop(key, None)
            self._last_confidence.pop(key, None)
            self._last_updated.pop(key, None)
        if stale:
            logger.info(f"ActionStateTracker: expired {len(stale)} stale state(s)")

    def has_changed(self, key: str, new_action: str) -> bool:
        # Also treat expired states as changed
        if key in self._last_updated:
            try:
                cutoff = datetime.now() - timedelta(hours=cfg.ACTION_STATE_EXPIRY_HOURS)
                if datetime.fromisoformat(self._last_updated[key]) < cutoff:
                    return True   # state is stale — treat as changed
            except Exception:
                return True
        return self._last_action.get(key) != new_action

    def update(self, key: str, action: str, confidence: float):
        now = datetime.now().isoformat()
        self._last_action[key]     = action
        self._last_confidence[key] = confidence
        self._last_updated[key]    = now
        db.set_state("action_state", {
            "actions":     self._last_action,
            "confidences": self._last_confidence,
            "updated_at":  self._last_updated
        })

    def get_last(self, key: str) -> Optional[str]:
        # Return None if state is stale
        if key in self._last_updated:
            try:
                cutoff = datetime.now() - timedelta(hours=cfg.ACTION_STATE_EXPIRY_HOURS)
                if datetime.fromisoformat(self._last_updated[key]) < cutoff:
                    return None
            except Exception:
                return None
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
    print(f"  ⚠  ONLY say y if your order has FILLED in ThinkorSwim.")
    print(f"  ⚠  Do NOT say y if the order is still Working/Pending.")
    print(f"  ⚠  Check for a green 'Filled' confirmation before proceeding.")
    response = timed_input(
        f"  Has your {ticker} order FILLED? (y/n, 60s timeout): ",
        timeout=60, default="n"
    )

    if response != "y":
        print(f"  Skipped — not tracking {ticker}")
        print(f"  (Come back and use manual_override_open once the order fills)")
        print(sep + "\n")
        return None

    # Ask for actual fill price
    print(f"  Enter your ACTUAL fill price from ThinkorSwim.")
    print(f"  (Check the Filled Orders tab for the exact price)")
    fill_str = timed_input(
        f"  Fill price (suggested ${suggested_entry}): ",
        timeout=60, default=str(suggested_entry)
    )

    try:
        fill_price = float(fill_str) if fill_str else suggested_entry
        if fill_price <= 0:
            print(f"  Invalid price (must be > 0) — using suggested ${suggested_entry}")
            fill_price = suggested_entry
    except ValueError:
        fill_price = suggested_entry
        print(f"  Could not parse — using suggested ${suggested_entry}")

    # Sanity check: warn if fill price is very different from suggested
    if suggested_entry and suggested_entry > 0:
        diff_pct = abs(fill_price - suggested_entry) / suggested_entry * 100
        if diff_pct > 20:
            print(f"  ⚠  Fill ${fill_price} is {diff_pct:.0f}% different from suggested ${suggested_entry}")
            confirm = timed_input(
                f"  Are you sure? (y/n): ",
                timeout=30, default="n"
            )
            if confirm != "y":
                print(f"  Cancelled — not tracking {ticker}")
                print(sep + "\n")
                return None

    # Open the position — bypass can_enter() since trade is already executed
    # User has already clicked the button in ThinkorSwim, we just need to record it
    # Calling open_position goes through can_enter which could block on cooldown/limits
    # Use db.insert_position directly so nothing can block a trade already made
    from database import insert_position as _insert_pos
    from datetime import datetime as _dt
    import json as _json

    trade_   = scanner_result.get("trade", {})
    pricing_ = scanner_result.get("pricing", {})
    vol_     = scanner_result.get("vol", {})
    main_    = trade_.get("main_leg", {})
    exp_     = main_.get("exp", trade_.get("exp", ""))
    entry_dte_ = None
    if exp_:
        try:
            entry_dte_ = (_dt.strptime(exp_, "%Y-%m-%d") - _dt.now()).days
        except Exception:
            pass

    short_  = trade_.get("short_leg", {})
    strategy_ = trade_.get("strategy", "")

    # Store short leg details in notes so spread P&L can be computed correctly
    # raw_scanner_data has full details but notes gives quick access
    if "SPREAD" in strategy_ and short_:
        notes_ = _json.dumps({
            "short_strike":      short_.get("strike"),
            "short_option_type": short_.get("option_type") or main_.get("option_type"),
            "spread": True
        })
    else:
        notes_ = "recorded via y/n prompt"

    _pos_data = {
        "ticker":           scanner_result.get("ticker", "?"),
        "strategy":         trade_.get("strategy"),
        "direction":        trade_.get("direction"),
        "option_type":      main_.get("option_type"),
        "strike":           main_.get("strike"),
        "expiration":       exp_,
        "entry_price":      fill_price,
        "entry_cost":       fill_price * 100 * pricing_.get("contracts", 1),
        "contracts":        pricing_.get("contracts", 1),
        "stop_price":       round(fill_price * (1 - cfg.STOP_LOSS_PCT), 2),
        "target_price":     round(fill_price * (1 + cfg.PROFIT_TARGET_PCT), 2),
        "entry_time":       _dt.now().isoformat(),
        "confluence_score": scanner_result.get("confluence", {}).get("score"),
        "entry_ivr":        vol_.get("ivr"),
        "entry_dte":        entry_dte_,
        "notes":            notes_,
        "raw_scanner_data": _json.dumps(scanner_result, default=str)
    }
    position_id = _insert_pos(_pos_data)
    if position_id:
        loaded = db.get_position_by_id(position_id)
        if loaded:
            tracker._open[position_id] = loaded
        else:
            logger.warning(f"Could not reload position {position_id} from DB after insert")
        stop_price   = round(fill_price * (1 - cfg.STOP_LOSS_PCT), 2)
        target_price = round(fill_price * (1 + cfg.PROFIT_TARGET_PCT), 2)
        print(f"  ✓ {ticker} tracked — entry=${fill_price} "
              f"stop=${stop_price} target=${target_price}")
        db.log_journal_event(
            "POSITION_OPENED", ticker=ticker, position_id=position_id,
            action="ENTER", confidence=confidence,
            reason_summary=f"User confirmed fill: {ticker} @ ${fill_price}",
            details={"fill_price": fill_price, "suggested": suggested_entry}
        )
    else:
        print(f"  Could not record position — check logs")

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
    print(f"  Check ThinkorSwim for the current bid/ask before deciding.")
    response = timed_input(
        f"  Close {ticker} position? (y/n, 60s timeout): ",
        timeout=60, default="n"
    )

    if response != "y":
        print(f"  Keeping {ticker} open — continuing to monitor")
        print(sep + "\n")
        return False

    # Ask for exit fill price
    fill_str = timed_input(
        f"  What did you close at? (suggested ${current_price:.2f}, press enter to use): ",
        timeout=60, default=str(round(current_price, 2))
    )

    try:
        exit_price = float(fill_str) if fill_str else current_price
        if exit_price <= 0:
            print(f"  Invalid price (must be > 0) — using suggested ${current_price:.2f}")
            exit_price = current_price
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
_PRICE_CACHE_TTL = 30  # seconds — refresh twice per tick for faster target/stop detection

def _fetch_option_mid(ticker: str, strike: float, expiration: str,
                      opt_type: str, api_key: str) -> Optional[float]:
    """
    Fetch mid price for a single option contract from Tradier.
    Returns None if not found or no valid bid/ask.
    """
    import requests as req
    try:
        r = req.get(
            "https://api.tradier.com/v1/markets/options/chains",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            params={"symbol": ticker, "expiration": expiration, "greeks": "false"},
            timeout=8
        )
        if r.status_code == 200:
            options = r.json().get("options", {}).get("option", []) or []
            for c in options:
                if (abs(float(c.get("strike", 0)) - float(strike)) < 0.01 and
                        c.get("option_type", "").lower() == opt_type.lower()):
                    bid  = float(c.get("bid", 0) or 0)
                    ask  = float(c.get("ask", 0) or 0)
                    last = float(c.get("last", 0) or 0)
                    if bid > 0 and ask > 0:
                        return round((bid + ask) / 2, 2)
                    if last > 0:
                        return round(last, 2)
    except Exception as e:
        logger.debug(f"Tradier fetch failed for {ticker} {strike} {opt_type}: {e}")
    return None


def get_current_option_price(position: Dict) -> Optional[float]:
    """
    Fetch the current net price for an open position.

    For spreads: fetches BOTH legs and returns long_mid - short_mid
    For single legs: fetches the one contract and returns its mid

    This is the critical function for P&L accuracy — must match ThinkorSwim.
    Caches for 30 seconds. Falls back to entry_price on failure.
    """
    import os

    position_id = position.get("id")

    # ── Check cache first
    if position_id and position_id in _price_cache:
        cached_price, cached_time = _price_cache[position_id]
        age = (datetime.now() - cached_time).total_seconds()
        if age < _PRICE_CACHE_TTL:
            return cached_price

    ticker      = position.get("ticker", "")
    strike      = position.get("strike")
    expiration  = position.get("expiration")
    entry_price = position.get("entry_price", 0)
    strategy    = (position.get("strategy") or "").upper()

    # Infer long leg option type from strategy
    stored_type = position.get("option_type") or ""
    if stored_type.lower() in ("call", "put"):
        opt_type = stored_type.lower()
    elif "BEAR" in strategy or "PUT" in strategy:
        opt_type = "put"
    elif "BULL" in strategy or "CALL" in strategy:
        opt_type = "call"
    else:
        opt_type = "put"

    if not ticker or not strike or not expiration:
        logger.warning(
            f"Cannot price position #{position_id}: missing ticker/strike/expiration"
        )
        return None

    # ── Check if this is a spread — read short leg from notes
    short_strike   = None
    short_opt_type = None
    is_spread      = "SPREAD" in strategy
    if is_spread:
        try:
            notes = json.loads(position.get("notes") or "{}")
            if notes.get("spread"):
                short_strike   = notes.get("short_strike")
                short_opt_type = notes.get("short_option_type") or opt_type
        except Exception:
            # Fall back to raw_scanner_data
            try:
                raw = json.loads(position.get("raw_scanner_data") or "{}")
                short_leg = raw.get("trade", {}).get("short_leg", {})
                if short_leg:
                    short_strike   = short_leg.get("strike")
                    short_opt_type = short_leg.get("option_type") or opt_type
            except Exception:
                pass

    def _cache_and_return(price: float) -> float:
        if position_id:
            _price_cache[position_id] = (price, datetime.now())
        return price

    api_key = os.environ.get("TRADIER_API_KEY", "")

    # ── Fetch long leg price
    long_mid = None
    if api_key:
        long_mid = _fetch_option_mid(ticker, strike, expiration, opt_type, api_key)

    # Fallback to yfinance for long leg
    if long_mid is None:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            chain = stock.option_chain(expiration)
            df  = chain.calls if opt_type == "call" else chain.puts
            row = df[abs(df["strike"] - float(strike)) < 0.01]
            if not row.empty:
                bid = float(row.iloc[0]["bid"] or 0)
                ask = float(row.iloc[0]["ask"] or 0)
                if bid > 0 and ask > 0:
                    long_mid = round((bid + ask) / 2, 2)
        except Exception as e:
            logger.debug(f"yfinance long leg failed for {ticker}: {e}")

    if long_mid is None:
        logger.warning(
            f"Could not fetch live option price for {ticker} {strike} {opt_type} "
            f"exp {expiration}; treating quote as stale"
        )
        return None

    # ── For spreads: fetch short leg and compute net value
    if is_spread and short_strike:
        short_mid = None
        if api_key:
            short_mid = _fetch_option_mid(
                ticker, short_strike, expiration, short_opt_type, api_key
            )
        if short_mid is None:
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                chain = stock.option_chain(expiration)
                df  = chain.calls if short_opt_type == "call" else chain.puts
                row = df[abs(df["strike"] - float(short_strike)) < 0.01]
                if not row.empty:
                    bid = float(row.iloc[0]["bid"] or 0)
                    ask = float(row.iloc[0]["ask"] or 0)
                    if bid > 0 and ask > 0:
                        short_mid = round((bid + ask) / 2, 2)
            except Exception as e:
                logger.debug(f"yfinance short leg failed for {ticker}: {e}")

        if short_mid is not None:
            # Spread value = long leg mid - short leg mid
            # This is what ThinkorSwim shows as the spread's current value
            spread_mid = round(long_mid - short_mid, 2)
            logger.debug(
                f"{ticker} spread: long={long_mid} short={short_mid} net={spread_mid}"
            )
            return _cache_and_return(max(spread_mid, 0.01))
        else:
            logger.debug(f"Could not fetch short leg for {ticker} — using long leg only")

    # ── Single leg or spread fallback: return long leg mid
    return _cache_and_return(long_mid)


# =============================================================================
#  TIER 2 FEATURES — EARNINGS / SECTOR / OI
# =============================================================================

def get_next_earnings_date(ticker: str) -> Optional[str]:
    """
    Return the next earnings date for a ticker as an ISO date string,
    or None if unknown or more than 60 days away.

    Uses Tradier fundamentals API first, yfinance as fallback.
    Results cached for 6 hours to avoid hammering the API.
    """
    import time

    # Check cache — refresh every 6 hours
    now_ts = time.time()
    if ticker in _earnings_cache_time:
        if now_ts - _earnings_cache_time[ticker] < 21600:  # 6 hours
            return _earnings_cache.get(ticker)

    date_str = None

    # ── Try Tradier fundamentals (beta endpoint)
    # Note: Tradier's fundamentals/calendars endpoint returns earnings dates
    # under tables.earnings.report_date — fiscal_year_end is NOT earnings date
    api_key = os.environ.get("TRADIER_API_KEY", "")
    if api_key:
        try:
            import requests as req
            r = req.get(
                "https://api.tradier.com/beta/markets/fundamentals/calendars",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json"
                },
                params={"symbols": ticker},
                timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                events = data if isinstance(data, list) else []
                today = datetime.now().strftime("%Y-%m-%d")
                for item in events:
                    tables = item.get("tables", {}) or {}
                    # Correct path: tables.earnings.report_date
                    earnings = tables.get("earnings", {}) or {}
                    report_date = earnings.get("report_date")
                    if report_date and report_date >= today:
                        date_str = report_date
                        break
                    # Also check corporate_calendars list
                    corp_cals = tables.get("corporate_calendars") or []
                    for cal in (corp_cals if isinstance(corp_cals, list) else []):
                        event_type = (cal.get("event") or "").lower()
                        cal_date   = cal.get("date") or cal.get("report_date")
                        if "earn" in event_type and cal_date and cal_date >= today:
                            date_str = cal_date
                            break
                    if date_str:
                        break
        except Exception as e:
            logger.debug(f"Tradier earnings fetch failed for {ticker}: {e}")

    # ── Fallback: yfinance
    if not date_str:
        try:
            import yfinance as yf
            from datetime import timezone
            t = yf.Ticker(ticker)
            dates = t.earnings_dates
            if dates is not None and not dates.empty:
                future = dates[dates.index > datetime.now(tz=timezone.utc)]
                if not future.empty:
                    # earnings_dates sorted descending — index[0] is nearest upcoming date
                    next_date = future.index[0]
                    date_str = next_date.strftime("%Y-%m-%d")
        except Exception as e:
            logger.debug(f"yfinance earnings fetch failed for {ticker}: {e}")

    # Cache result
    _earnings_cache[ticker] = date_str
    _earnings_cache_time[ticker] = now_ts
    return date_str


def check_earnings_proximity(ticker: str) -> tuple:
    """
    Check if ticker has earnings coming up soon.
    Returns (days_to_earnings: int or None, warning: str or None)

    days_to_earnings = None means no upcoming earnings found.
    """
    if not cfg.EARNINGS_CHECK_ENABLED:
        return None, None

    date_str = get_next_earnings_date(ticker)
    if not date_str:
        return None, None

    try:
        exp_dt = datetime.strptime(date_str, "%Y-%m-%d")
        days = (exp_dt - datetime.now()).days
        if days < 0:
            return None, None   # already passed
        if days <= cfg.EARNINGS_BLOCK_DAYS:
            return days, f"EARNINGS IN {days}d — ENTRY BLOCKED (too close to report)"
        if days <= cfg.EARNINGS_WARN_DAYS:
            return days, f"EARNINGS IN {days}d — IV crush risk after report"
        return days, None
    except Exception:
        return None, None


def check_sector_correlation(
    ticker: str,
    direction: str,
    tracker: PositionTracker
) -> tuple:
    """
    Check if adding this position would create too much sector/direction concentration.
    Returns (blocked: bool, warning: str or None)
    """
    if not cfg.SECTOR_CORRELATION_ENABLED:
        return False, None

    sector = cfg.SECTOR_MAP.get(ticker.upper(), "unknown")
    open_positions = list(tracker.open_positions.values())

    # Count same-sector positions
    same_sector = [
        p for p in open_positions
        if cfg.SECTOR_MAP.get(p.get("ticker", "").upper(), "?") == sector
        and sector != "unknown"
    ]

    # Count same-direction positions
    same_direction = [
        p for p in open_positions
        if (p.get("direction") or "").upper() == (direction or "").upper()
        and direction
    ]

    warnings = []

    if len(same_sector) >= cfg.MAX_SAME_SECTOR_POSITIONS:
        tickers = [p["ticker"] for p in same_sector]
        warnings.append(
            f"Sector concentration: already {len(same_sector)} {sector} positions "
            f"({', '.join(tickers)}) — adding {ticker} increases correlated risk"
        )

    if len(same_direction) >= cfg.MAX_SAME_DIRECTION_POSITIONS:
        tickers = [p["ticker"] for p in same_direction]
        warnings.append(
            f"Direction concentration: {len(same_direction)} {direction} positions already open "
            f"({', '.join(tickers)}) — correlated market exposure"
        )

    if warnings:
        return False, " | ".join(warnings)   # warn but don't block
    return False, None


def check_oi_confirms_flow(scanner_result: Dict) -> tuple:
    """
    Compare current OI to previous scan OI to determine if flow
    represents opening or closing positions.

    Opening flow (OI increasing) = new position being established = stronger signal
    Closing flow (OI flat/decreasing) = existing position being exited = weaker signal

    Returns (is_opening_flow: bool, note: str)
    """
    if not cfg.OI_CHANGE_ENABLED:
        return True, ""

    trade = scanner_result.get("trade", {})
    main  = trade.get("main_leg", {})
    ticker = scanner_result.get("ticker", "")
    strike = main.get("strike")
    exp    = main.get("exp") or trade.get("exp", "")
    opt_type = main.get("option_type", "put")
    oi     = main.get("open_interest") or main.get("oi")

    if not all([ticker, strike, exp, oi]):
        return True, ""   # can't check — assume valid

    cache_key = f"{ticker}:{strike}:{exp}:{opt_type}"
    prev_oi   = _oi_cache.get(cache_key)

    # Update cache with current OI
    _oi_cache[cache_key] = int(oi)

    if prev_oi is None:
        return True, "OI baseline established (first scan)"

    oi_change = int(oi) - prev_oi
    pct_change = (oi_change / max(prev_oi, 1)) * 100

    if oi_change > 0:
        return True, f"OI +{oi_change:,} ({pct_change:+.1f}%) — confirms opening flow"
    elif oi_change == 0:
        return not cfg.OI_INCREASE_REQUIRED,                f"OI unchanged — may be closing flow (existing position exiting)"
    else:
        return False,                f"OI {oi_change:,} ({pct_change:+.1f}%) — likely CLOSING flow, not new position"


def _record_broker_order(
    oid: str,
    order_kind: str,
    status: str,
    ticker: str,
    side: str,
    strategy: str,
    option_type: str,
    strike: float,
    expiration: str,
    quantity: int,
    limit_price: float,
    client_order_id: str = None,
    position_id: int = None,
    raw_payload: dict = None,
    notes: str = "",
):
    """Persist broker order lifecycle state without interrupting trading flow."""
    if not oid:
        return
    try:
        db.upsert_order({
            "broker_order_id": oid,
            "client_order_id": client_order_id,
            "order_kind": order_kind,
            "status": status,
            "ticker": ticker,
            "position_id": position_id,
            "side": side,
            "strategy": strategy,
            "option_type": option_type,
            "strike": strike,
            "expiration": expiration,
            "quantity": quantity,
            "limit_price": limit_price,
            "raw_payload": raw_payload or {},
            "notes": notes,
        })
    except Exception as e:
        logger.warning(f"Order state write failed for {oid}: {e}")


def _update_broker_order(oid: str, status: str, filled_price: float = None,
                         position_id: int = None, raw_payload: dict = None,
                         notes: str = None):
    if not oid:
        return
    try:
        db.update_order_status(
            oid,
            status=status,
            filled_price=filled_price,
            position_id=position_id,
            raw_payload=raw_payload,
            notes=notes,
        )
    except Exception as e:
        logger.warning(f"Order state update failed for {oid}: {e}")


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

    def _valid_scanner_result(r: Dict) -> bool:
        trade = r.get("trade", {})
        if r.get("error"):
            return False
        if "THEORETICAL" in str(trade.get("data_quality", "")).upper():
            return False
        if trade.get("short_leg_synthetic"):
            return False
        if r.get("confluence", {}).get("score", 0) < cfg.MIN_CONFLUENCE_SCORE_TO_ENTER:
            return False
        if not trade.get("main_leg", {}).get("strike"):
            return False
        if (r.get("pricing", {}).get("entry") or 0) < 0.50:
            return False
        return True

    valid = [r for r in results if _valid_scanner_result(r)]

    logger.info(f"Scanner complete: {len(results)} scanned, {len(valid)} meet threshold")
    return sorted(valid, key=lambda r: r.get("confluence", {}).get("score", 0), reverse=True)


def evaluate_open_positions(
    tracker: PositionTracker,
    agent: DecisionAgent,
    action_state: ActionStateTracker,
    scanner_results: List[Dict],
    regime: Dict,
    auto_mode: bool = False,
    bankroll: Optional[list] = None,
    live_paper: bool = False,
    pending_exit_orders: dict = None
) -> List[Dict]:
    """
    Evaluate each open position. Apply hard rules first, then agent scoring.
    In auto_mode: exits are executed automatically with no user prompt.
    In bankroll mode: proceeds from closes are credited back to the bankroll pool.
    In live_paper: closing orders sent to Tradier sandbox for real fills.
    Returns list of action dicts taken this tick.
    """
    actions_taken = []

    # Build a lookup from ticker → latest scanner result
    scanner_by_ticker = {
        r.get("ticker", "").upper(): r for r in scanner_results
    }

    for position_id, position in list(tracker.open_positions.items()):
        ticker = position["ticker"].upper()

        if pending_exit_orders and any(
            int(v.get("position_id", -1)) == int(position_id)
            for v in pending_exit_orders.values()
        ):
            logger.info(f"  {ticker} #{position_id} already has a pending exit order — waiting for broker fill")
            continue

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
        if current_option_price is None or current_option_price <= 0:
            logger.warning(
                f"  {ticker} #{position_id}: stale/missing option quote — "
                "skipping hard exits and agent scoring this tick"
            )
            db.log_journal_event(
                "STALE_QUOTE",
                ticker=ticker,
                position_id=position_id,
                reason_summary="Missing live option quote; skipped exit evaluation",
                details={"position": position}
            )
            continue

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

            # ── Get exit price — live paper, auto, or manual
            sep = "=" * 64
            if auto_mode:
                confirmed_exit_price = current_option_price

                # Place real closing order if in live-paper mode
                if live_paper and BROKER_AVAILABLE and broker._is_configured():
                    is_spread = "SPREAD" in (position.get("strategy") or "")
                    try:
                        notes_data = json.loads(position.get("notes") or "{}")
                    except Exception:
                        notes_data = {}

                    if is_spread and notes_data.get("short_strike"):
                        # Use market order for spread close — always want out on stop/target
                        ok, oid, _ = broker.place_spread_order(
                            ticker       = ticker,
                            expiration   = position.get("expiration", ""),
                            option_type  = position.get("option_type", "call"),
                            long_strike  = position.get("strike", 0),
                            short_strike = notes_data["short_strike"],
                            side         = "sell",
                            quantity     = position.get("contracts", 1),
                            net_limit_price = current_option_price,
                            market_order = True
                        )
                    else:
                        # Use market order for single-leg close — always want out on stop/target
                        ok, oid, _ = broker.place_option_order(
                            ticker       = ticker,
                            expiration   = position.get("expiration", ""),
                            option_type  = position.get("option_type", "call"),
                            strike       = position.get("strike", 0),
                            side         = "sell_to_close",
                            quantity     = position.get("contracts", 1),
                            limit_price  = current_option_price,
                            market_order = True
                        )
                    if ok:
                        _record_broker_order(
                            oid=oid,
                            order_kind="EXIT",
                            status="SUBMITTED",
                            ticker=ticker,
                            side="sell_to_close",
                            strategy=position.get("strategy"),
                            option_type=position.get("option_type", "call"),
                            strike=position.get("strike", 0),
                            expiration=position.get("expiration", ""),
                            quantity=position.get("contracts", 1),
                            limit_price=current_option_price,
                            position_id=position_id,
                            raw_payload={"exit_reason": hard_reason, "market_order": True},
                            notes="hard_exit",
                        )
                        logger.info(f"  Closing order placed for {ticker} order={oid} — waiting for broker fill confirmation")
                        if pending_exit_orders is not None:
                            pending_exit_orders[oid] = {
                                "position_id": position_id,
                                "ticker": ticker,
                                "exit_reason": hard_reason,
                                "submitted_at": datetime.now().isoformat(),
                                "estimated_exit_price": current_option_price,
                                "position": position,
                                "regime": regime,
                                "snapshot": snapshot,
                            }
                            save_pending_exit_orders(pending_exit_orders)
                        action_state.update(str(position_id), "EXIT_PENDING", 1.0)
                        actions_taken.append({
                            "type": "EXIT_SUBMITTED", "ticker": ticker,
                            "reason": hard_reason, "position_id": position_id,
                            "order_id": oid
                        })
                        continue
                    logger.error(f"  Closing order failed for {ticker}; position remains open in tracker")
                    actions_taken.append({
                        "type": "EXIT_ORDER_FAILED", "ticker": ticker,
                        "reason": hard_reason, "position_id": position_id
                    })
                    continue

                print(f"\n{sep}")
                print(f"  [AUTO] {hard_reason} — closing {ticker} @ ${confirmed_exit_price:.2f}")
                print(sep + "\n")
            else:
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

            # ── Credit bankroll with proceeds
            if bankroll is not None:
                proceeds = confirmed_exit_price * 100 * position.get("contracts", 1)
                bankroll[0] = round(bankroll[0] + proceeds, 2)
                db.set_state("bankroll_remaining", bankroll[0])
                pnl_amt = result.get("realized_pnl", 0) or 0
                sign = "+" if pnl_amt >= 0 else ""
                print(f"  [BANKROLL] {hard_reason} closed — proceeds ${proceeds:.2f} "
                      f"returned. P&L: {sign}${pnl_amt:.2f}. "
                      f"Bankroll now: ${bankroll[0]:,.2f}")

            # Update agent — inject regime into entry snapshot so
            # regime_score feature is populated correctly
            entry_snapshot = json.loads(
                position.get("raw_scanner_data") or "{}"
            )
            entry_snapshot["regime_data"] = regime
            agent.update_on_close(
                position=position,
                exit_reason=hard_reason,
                realized_r=realized_r,
                entry_market_snapshot=build_market_snapshot(entry_snapshot),
                exit_market_snapshot=snapshot,
                ticks_held=_ticks_held(position),
                rolling_drawdown=_rolling_drawdown()
            )

            _clear_price_cache(position_id)
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
                if secs_held < 600:  # 10 minute grace period — matches hard exit rule
                    in_grace_period = True
            except Exception:
                pass

        if in_grace_period:
            action     = "HOLD"
            confidence = 0.0
            reasons    = ["Grace period — position too new to evaluate (< 10 min)"]
        else:
            action, confidence, reasons = agent.score_exit(position, snapshot, ticks)

        # Persist tick snapshot to DB — skip if position was deleted externally
        if not db.get_position_by_id(position_id):
            logger.warning(f"Position #{position_id} no longer in DB — removing from tracker")
            tracker._open.pop(position_id, None)
            _clear_price_cache(position_id)
            continue
        try:
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
                "feature_vector": _json_dumps([]),
                "agent_action":   action,
                "agent_confidence": confidence,
                "agent_reasons":  _json_dumps(reasons[:5])
            })
        except Exception as snap_err:
            if "FOREIGN KEY" in str(snap_err):
                # Position was deleted externally (--delete while loop running)
                # Remove from in-memory tracker so loop stops evaluating it
                logger.warning(
                    f"Position #{position_id} ({ticker}) was deleted externally — "
                    f"removing from in-memory tracker. Restart loop to fully clean up."
                )
                tracker._open.pop(position_id, None)
                _clear_price_cache(position_id)
                continue
            logger.error(f"tick_snapshot insert failed: {snap_err}")

        # Alert if action changed and confidence is high enough
        key = str(position_id)
        changed = action_state.has_changed(key, action)

        # Debug: always print exit scoring for visibility
        logger.info(f"  {ticker} #{position_id} exit scoring: action={action} conf={confidence:.3f} changed={changed} score_threshold={cfg.EXIT_CONFIDENCE_THRESHOLD}")

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

            if auto_mode:
                # Determine exit price — live paper gets real fill, otherwise use mid
                exit_price = current_option_price
                if live_paper and BROKER_AVAILABLE and broker._is_configured():
                    is_spread = "SPREAD" in (position.get("strategy") or "")
                    try:
                        notes_data = json.loads(position.get("notes") or "{}")
                    except Exception:
                        notes_data = {}

                    if is_spread and notes_data.get("short_strike"):
                        ok, oid, _ = broker.place_spread_order(
                            ticker       = ticker,
                            expiration   = position.get("expiration", ""),
                            option_type  = position.get("option_type", "call"),
                            long_strike  = position.get("strike", 0),
                            short_strike = notes_data["short_strike"],
                            side         = "sell",
                            quantity     = position.get("contracts", 1),
                            net_limit_price = current_option_price
                        )
                    else:
                        ok, oid, _ = broker.place_option_order(
                            ticker       = ticker,
                            expiration   = position.get("expiration", ""),
                            option_type  = position.get("option_type", "call"),
                            strike       = position.get("strike", 0),
                            side         = "sell_to_close",
                            quantity     = position.get("contracts", 1),
                            limit_price  = current_option_price,
                            market_order = True
                        )
                    if ok:
                        _record_broker_order(
                            oid=oid,
                            order_kind="EXIT",
                            status="SUBMITTED",
                            ticker=ticker,
                            side="sell_to_close",
                            strategy=position.get("strategy"),
                            option_type=position.get("option_type", "call"),
                            strike=position.get("strike", 0),
                            expiration=position.get("expiration", ""),
                            quantity=position.get("contracts", 1),
                            limit_price=current_option_price,
                            position_id=position_id,
                            raw_payload={"exit_reason": "AGENT_EXIT", "market_order": True},
                            notes="agent_exit",
                        )
                        logger.info(f"  Exit order placed for {ticker} order={oid} — waiting for broker fill confirmation")
                        if pending_exit_orders is not None:
                            pending_exit_orders[oid] = {
                                "position_id": position_id,
                                "ticker": ticker,
                                "exit_reason": "AGENT_EXIT",
                                "submitted_at": datetime.now().isoformat(),
                                "estimated_exit_price": current_option_price,
                                "position": position,
                                "regime": regime,
                                "snapshot": snapshot,
                            }
                            save_pending_exit_orders(pending_exit_orders)
                        print(f"  [LIVE-PAPER] EXIT submitted for {ticker} order={oid} @ est ${current_option_price:.2f}")
                        action_state.update(str(position_id), "EXIT_PENDING", confidence)
                        actions_taken.append({
                            "type": "EXIT_SUBMITTED", "ticker": ticker,
                            "reason": "AGENT_EXIT", "position_id": position_id,
                            "order_id": oid
                        })
                        continue
                    logger.error(f"  Exit order failed for {ticker}; position remains open in tracker")
                    actions_taken.append({
                        "type": "EXIT_ORDER_FAILED", "ticker": ticker,
                        "reason": "AGENT_EXIT", "position_id": position_id
                    })
                    continue

                pnl  = tracker.unrealized_pnl(position_id, exit_price)
                r_   = tracker.unrealized_r(position_id, exit_price)
                sign = "+" if pnl >= 0 else ""
                mode_tag = "[LIVE-PAPER]" if live_paper else "[AUTO]"
                print(f"  {mode_tag} EXIT {ticker} @ ${exit_price:.2f} "
                      f"P&L: {sign}${pnl:.2f} ({r_:+.2f}R)")
                closed = True
                tracker.close_position(position_id, exit_price, "AGENT_EXIT")
                # Credit bankroll with proceeds
                if bankroll is not None:
                    proceeds = exit_price * 100 * position.get("contracts", 1)
                    bankroll[0] = round(bankroll[0] + proceeds, 2)
                    db.set_state("bankroll_remaining", bankroll[0])
                    print(f"  [BANKROLL] Proceeds ${proceeds:.2f} returned. "
                          f"Bankroll now: ${bankroll[0]:,.2f}")
                # Note: _clear_price_cache called in the "if closed:" block below
            else:
                # ── Ask user if they want to close this position
                closed = ask_close_position(
                    position_id=position_id,
                    position=position,
                    tracker=tracker,
                    current_price=current_option_price,
                    confidence=confidence,
                    reason="AGENT_EXIT"
                )

            # If user closed it, update agent using ACTUAL realized_r from close
            if closed:
                _clear_price_cache(position_id)
                # ask_close_position already called tracker.close_position()
                # Get the realized_r that was recorded — not the snapshot estimate
                closed_pos = db.get_position_by_id(position_id)
                actual_realized_r = (
                    closed_pos.get("realized_r", 0)
                    if closed_pos else snapshot.get("unrealized_r", 0)
                )
                entry_snap = json.loads(position.get("raw_scanner_data") or "{}")
                entry_snap["regime_data"] = regime
                agent.update_on_close(
                    position=position,
                    exit_reason="AGENT_EXIT",
                    realized_r=actual_realized_r,
                    entry_market_snapshot=build_market_snapshot(entry_snap),
                    exit_market_snapshot=snapshot,
                    ticks_held=_ticks_held(position),
                    rolling_drawdown=_rolling_drawdown()
                )
                action_state.update(str(position_id), "EXIT", confidence)

            # Only insert recommendation if position still exists in DB
            if db.get_position_by_id(position_id):
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
                    "acted_upon":     1 if closed else 0,
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
    regime: Dict,
    auto_mode: bool = False,
    bankroll: Optional[list] = None,
    live_paper: bool = False,
    pending_orders: dict = None
) -> List[Dict]:
    """
    Evaluate scanner recommendations for potential new entries.
    In auto_mode: positions are opened automatically at suggested price.
    In bankroll mode: entries are funded from the bankroll pool (1 contract max).
    In live_paper: entry orders sent to Tradier sandbox for real fills.
    No position limit in auto_mode/bankroll_mode.
    Hard rules checked first, then agent scores ENTER vs WAIT.
    """
    actions_taken = []
    pending_entry_cost = 0.0
    if pending_orders:
        for pdata in pending_orders.values():
            pending_entry_cost += (
                (pdata.get("limit_price") or pdata.get("entry_price") or 0)
                * 100
                * (pdata.get("contracts") or 1)
            )

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

        # Skip if ticker already has a pending unfilled order
        if pending_orders and any(
            v.get("ticker", "").upper() == ticker for v in pending_orders.values()
        ):
            logger.debug(f"  {ticker} has pending order — skipping alert")
            continue

        # Skip if on cooldown
        if db.is_on_cooldown(ticker):
            cd = db.get_cooldown(ticker)
            if DEBUG_MODE:
                logger.debug(f"  {ticker} on cooldown until {cd.get('cooldown_until', '?')[:16]}")
            continue

        allowed, risk_reason = tracker.can_enter(ticker)
        if not allowed:
            logger.info(f"  {ticker} blocked by risk gate — {risk_reason}")
            continue

        # Position limit and capital check
        if bankroll is not None:
            # Bankroll mode — check if pool can cover the next premium (1 contract)
            next_premium = (scanner_result.get("pricing", {}).get("entry", 0) or 0) * 100
            if next_premium <= 0:
                continue
            available_bankroll = bankroll[0] - pending_entry_cost
            if available_bankroll < next_premium:
                logger.info(
                    f"[BANKROLL] Insufficient funds after pending orders — "
                    f"need ${next_premium:.2f}, have ${available_bankroll:.2f} "
                    f"(${pending_entry_cost:.2f} reserved) — skipping {ticker}"
                )
                continue
        elif auto_mode:
            # Auto mode — position and capital limits
            if tracker.open_count >= cfg.AUTO_MAX_POSITIONS:
                logger.info(f"[AUTO] Max positions ({cfg.AUTO_MAX_POSITIONS}) reached — skipping {ticker}")
                break
            total_deployed = sum(
                p.get("entry_cost", 0) for p in tracker.open_positions.values()
            )
            total_deployed += pending_entry_cost
            max_deploy = cfg.ACCOUNT_SIZE * cfg.AUTO_MAX_CAPITAL_PCT
            next_cost  = (scanner_result.get("pricing", {}).get("entry", 0) or 0) * 100
            if total_deployed + next_cost > max_deploy:
                logger.info(
                    f"[AUTO] Capital limit reached — deployed ${total_deployed:.0f} / "
                    f"${max_deploy:.0f} max — skipping {ticker}"
                )
                continue
        else:
            if tracker.open_count >= cfg.MAX_CONCURRENT_POSITIONS:
                logger.info(f"Max positions ({cfg.MAX_CONCURRENT_POSITIONS}) reached — skipping {ticker}")
                break

        # Skip if outside market hours entry window
        if not is_market_hours_for_entry():
            logger.info(f"  {ticker} skipped — outside market hours entry window (10am-3:30pm ET)")
            continue

        # ── Earnings check
        days_to_earnings, earnings_warning = check_earnings_proximity(ticker)
        if earnings_warning and "BLOCKED" in earnings_warning:
            logger.info(f"  {ticker} BLOCKED — {earnings_warning}")
            continue

        # ── OI change detection — validate flow is opening not closing
        oi_valid, oi_note = check_oi_confirms_flow(scanner_result)

        # ── Sector / direction correlation check (warn only, not block)
        direction = scanner_result.get("trade", {}).get("direction", "")
        trade_info = scanner_result.get("trade", {})
        strategy_name = (trade_info.get("strategy") or "").upper()
        tech_info = scanner_result.get("tech", {})
        sector_info = scanner_result.get("sector", {})
        broad_market = sector_info.get("broad_market", "UNKNOWN")
        if (
            direction == "BULLISH"
            and ("CALL" in strategy_name or "BULL" in strategy_name)
            and broad_market == "BEARISH"
        ):
            logger.info(f"  {ticker} blocked — bullish call setup against bearish SPY/QQQ tape")
            continue
        if (
            direction == "BULLISH"
            and ("CALL" in strategy_name or "BULL" in strategy_name)
            and tech_info.get("above_vwap") is False
        ):
            logger.info(f"  {ticker} blocked — bullish call setup while ticker is below VWAP")
            continue

        _, correlation_warning = check_sector_correlation(ticker, direction, tracker)
        if correlation_warning and (auto_mode or live_paper):
            logger.info(f"  {ticker} blocked — {correlation_warning}")
            continue

        snapshot = build_market_snapshot(scanner_result)

        # Agent scores the entry
        action, confidence, reasons = agent.score_entry(scanner_result, snapshot)

        # Inject tier 2 signals into reasons so user sees them in the alert
        tier2_notes = []
        if earnings_warning:
            tier2_notes.append(f"⚠ {earnings_warning}")
        if days_to_earnings and not earnings_warning:
            tier2_notes.append(f"Earnings in {days_to_earnings}d")
        if oi_note and oi_note != "OI baseline established (first scan)":
            tier2_notes.append(f"OI: {oi_note}")
        if not oi_valid:
            # Closing flow — reduce confidence
            confidence = round(confidence * 0.75, 3)
            tier2_notes.append("⚠ OI suggests closing flow — confidence reduced 25%")
        flow = scanner_result.get("flow", [])
        if flow and flow[0].get("dir_confidence") == "LOW":
            confidence = round(confidence * 0.85, 3)
            tier2_notes.append("⚠ Flow direction ambiguous — confidence reduced 15%")
        if correlation_warning:
            tier2_notes.append(f"⚠ {correlation_warning}")

        if tier2_notes:
            reasons = tier2_notes + reasons

        trade = scanner_result.get("trade", {})
        pricing = scanner_result.get("pricing", {})
        main_leg = trade.get("main_leg", {})
        flow_conf = flow[0].get("dir_confidence") if flow else "none"
        regime_name = snapshot.get("regime")
        candidate_eval_id = None
        try:
            candidate_eval_id = db.insert_candidate_evaluation({
                "timestamp": snapshot["timestamp"],
                "ticker": ticker,
                "strategy": trade.get("strategy"),
                "direction": trade.get("direction"),
                "strike": main_leg.get("strike"),
                "expiration": main_leg.get("exp") or trade.get("exp"),
                "action": action,
                "confidence": confidence,
                "confluence_score": scanner_result.get("confluence", {}).get("score"),
                "flow_confidence": flow_conf,
                "ivr": scanner_result.get("vol", {}).get("ivr"),
                "regime": regime_name,
                "above_vwap": 1 if snapshot.get("above_vwap") else 0,
                "entry_price": pricing.get("entry"),
                "reasons": reasons[:8],
                "market_snapshot": snapshot,
                "raw_scanner_data": scanner_result,
            })
        except Exception as e:
            logger.warning(f"Candidate evaluation log failed for {ticker}: {e}")

        key = f"entry_{ticker}"
        changed = action_state.has_changed(key, action)
        _just_tracked = False   # initialize here — set True if user confirms fill

        # Suppress alerts during exploration ticks — random confidence
        # should not fire user-facing notifications
        is_explore_tick = getattr(agent, "_last_was_explore", False)

        should_notify = (
            changed and
            action == "ENTER" and
            confidence >= cfg.NOTIFY_CONFIDENCE_THRESHOLD and
            not is_explore_tick
        )

        if should_notify:
            short_leg = trade.get("short_leg", {})
            strategy  = trade.get("strategy", "")
            exp       = main_leg.get("exp") or trade.get("exp")

            # Infer option type from strategy name — more reliable than main_leg data
            # BEAR strategies use puts, BULL strategies use calls
            # Fall back to main_leg option_type, then default to put
            if "BEAR" in strategy.upper() or "PUT" in strategy.upper():
                opt_label = "PUT"
            elif "BULL" in strategy.upper() or "CALL" in strategy.upper():
                opt_label = "CALL"
            else:
                opt_label = (main_leg.get("option_type") or "put").upper()

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

            # ── Enter position — auto or manual
            if auto_mode:
                # Auto-enter at scanner suggested price
                pricing_  = scanner_result.get("pricing", {})
                trade_    = scanner_result.get("trade", {})
                main_     = trade_.get("main_leg", {})
                short_    = trade_.get("short_leg", {})
                strategy_ = trade_.get("strategy", "")
                # Use mid price as limit; live price refresh happens before order placement
                fill_price_ = pricing_.get("entry", main_.get("mid", 0)) or 0
                exp_      = main_.get("exp", trade_.get("exp", ""))
                vol_      = scanner_result.get("vol", {})

                if fill_price_ > 0:
                    from datetime import datetime as _dt
                    import json as _json
                    from database import insert_position as _ins

                    # Bankroll mode always uses 1 contract
                    contracts_ = 1 if bankroll is not None else pricing_.get("contracts", 1)
                    entry_cost_ = fill_price_ * 100 * contracts_

                    short_notes = _json.dumps({
                        "short_strike":      short_.get("strike"),
                        "short_option_type": short_.get("option_type") or main_.get("option_type"),
                        "spread": True
                    }) if "SPREAD" in strategy_ and short_ else (
                        "bankroll mode entry" if bankroll is not None else "auto mode entry"
                    )

                    entry_dte_ = None
                    try:
                        entry_dte_ = (_dt.strptime(exp_, "%Y-%m-%d") - _dt.now()).days
                    except Exception:
                        pass

                    _pos_data = {
                        "ticker":           ticker,
                        "strategy":         strategy_,
                        "direction":        trade_.get("direction"),
                        "option_type":      main_.get("option_type"),
                        "strike":           main_.get("strike"),
                        "expiration":       exp_,
                        "entry_price":      fill_price_,
                        "entry_cost":       entry_cost_,
                        "contracts":        contracts_,
                        "stop_price":       round(fill_price_ * (1 - cfg.STOP_LOSS_PCT), 2),
                        "target_price":     round(fill_price_ * (1 + cfg.PROFIT_TARGET_PCT), 2),
                        "entry_time":       _dt.now().isoformat(),
                        "confluence_score": scanner_result.get("confluence", {}).get("score"),
                        "entry_ivr":        vol_.get("ivr"),
                        "entry_dte":        entry_dte_,
                        "notes":            short_notes,
                        "raw_scanner_data": _json.dumps(scanner_result, default=str)
                    }
                    # ── Live-paper mode: place order first, only write to DB on confirmed fill
                    actual_fill = fill_price_
                    position_id = None
                    oid = None

                    if live_paper and BROKER_AVAILABLE and broker._is_configured():
                        # Refresh price from Tradier immediately before order — scanner data may be stale
                        if SCANNER_AVAILABLE:
                            try:
                                occ_symbol = broker.build_option_symbol(
                                    ticker, exp_, main_.get("option_type", "call"), main_.get("strike", 0)
                                )
                                fresh = get_live_option_quote(occ_symbol) if get_live_option_quote else {}
                                # run_live.py always uses mid for price discipline
                                use_price = fresh.get("mid", 0)
                                if use_price > 0:
                                    stale_price = fill_price_
                                    # Sanity check — if refreshed price diverges >40% from scanner price, skip trade
                                    if stale_price > 0:
                                        divergence = abs(use_price - stale_price) / stale_price
                                        if divergence > 0.40:
                                            logger.warning(f"  {ticker} price divergence too large (scanner=${stale_price:.2f} live=${use_price:.2f} diff={divergence:.0%}) — skipping trade")
                                            _just_tracked = False
                                            continue
                                    fill_price_ = use_price
                                    logger.info(f"  {ticker} price refreshed: ${stale_price:.2f} -> mid ${fill_price_:.2f} (bid=${fresh['bid']:.2f} ask=${fresh['ask']:.2f})")
                                    # Re-check bankroll against actual live price
                                    if bankroll is not None and fill_price_ * 100 > bankroll[0]:
                                        logger.info(f"  [BANKROLL] Insufficient funds after price refresh — need ${fill_price_*100:.2f}, have ${bankroll[0]:.2f} — skipping {ticker}")
                                        _just_tracked = False
                                        continue
                            except Exception as e:
                                logger.warning(f"  {ticker} price refresh failed: {e} — using cached price")

                    if live_paper and BROKER_AVAILABLE and broker._is_configured():
                        # Check for existing open order on this ticker before placing
                        open_orders = broker.get_open_orders()
                        already_pending = any(ticker in o.get("symbol", "") for o in open_orders)
                        if already_pending:
                            logger.info(f"  {ticker} already has open order at Alpaca — skipping duplicate entry")
                            _just_tracked = False
                        else:
                            unique_tag = f"rl_{ticker}_{int(time.time())}"
                            is_spread = "SPREAD" in strategy_
                            if is_spread and short_:
                                ok, oid, _ = broker.place_spread_order(
                                    ticker          = ticker,
                                    expiration      = exp_,
                                    option_type     = main_.get("option_type", "call"),
                                    long_strike     = main_.get("strike", 0),
                                    short_strike    = short_.get("strike", 0),
                                    side            = "buy",
                                    quantity        = contracts_,
                                    net_limit_price = fill_price_,
                                    tag             = unique_tag
                                )
                            else:
                                ok, oid, _ = broker.place_option_order(
                                    ticker      = ticker,
                                    expiration  = exp_,
                                    option_type = main_.get("option_type", "call"),
                                    strike      = main_.get("strike", 0),
                                    side        = "buy_to_open",
                                    quantity    = contracts_,
                                    limit_price = fill_price_,
                                    tag         = unique_tag
                                )

                            if ok:
                                _record_broker_order(
                                    oid=oid,
                                    order_kind="ENTRY",
                                    status="SUBMITTED",
                                    ticker=ticker,
                                    side="buy_to_open",
                                    strategy=strategy_,
                                    option_type=main_.get("option_type", "call"),
                                    strike=main_.get("strike", 0),
                                    expiration=exp_,
                                    quantity=contracts_,
                                    limit_price=fill_price_,
                                    client_order_id=unique_tag,
                                    raw_payload={
                                        "is_spread": is_spread,
                                        "short_strike": short_.get("strike") if short_ else None,
                                        "candidate_evaluation_id": candidate_eval_id,
                                    },
                                    notes="entry_submitted",
                                )
                                # Order placed — queue for reconciliation, don't block the loop
                                print(f"  [LIVE-PAPER] Order {oid} placed for {ticker} @ ${fill_price_:.2f} — queued for reconciliation")
                                if pending_orders is not None:
                                    pdata = _pos_data.copy()
                                    pdata["limit_price"] = fill_price_  # actual placed limit, not cached scanner price
                                    pdata["candidate_evaluation_id"] = candidate_eval_id
                                    pending_orders[oid] = pdata
                                    save_pending_orders(pending_orders)
                            else:
                                print(f"  [LIVE-PAPER] Order placement failed for {ticker} — not tracked")

                    else:
                        # Non-live-paper mode: write to DB immediately as before
                        position_id = _ins(_pos_data)
                        if position_id:
                            loaded = db.get_position_by_id(position_id)
                            if loaded:
                                tracker._open[position_id] = loaded

                        # Deduct from bankroll
                        if bankroll is not None:
                            bankroll[0] = round(bankroll[0] - (actual_fill * 100 * contracts_), 2)
                            db.set_state("bankroll_remaining", bankroll[0])
                            print(f"  [BANKROLL] ENTER {ticker} #{position_id} @ ${actual_fill:.2f} "
                                  f"({strategy_}) cost=${actual_fill*100*contracts_:.2f} "
                                  f"remaining=${bankroll[0]:,.2f}")
                        elif not live_paper:
                            print(f"  [AUTO] ENTER {ticker} #{position_id} @ ${actual_fill:.2f} "
                                  f"({strategy_}) conf={confidence:.0%}")

                        db.log_journal_event(
                            "POSITION_OPENED", ticker=ticker, position_id=position_id,
                            action="ENTER", confidence=confidence,
                            reason_summary=f"[{'BANKROLL' if bankroll is not None else 'AUTO'}] {ticker} @ ${fill_price_:.2f}",
                            details={"fill_price": fill_price_, "auto": True,
                                     "bankroll_remaining": bankroll[0] if bankroll is not None else None}
                        )
                else:
                    position_id = None
                    logger.warning(f"  [AUTO] {ticker} — no valid entry price, skipping")
                _just_tracked = position_id is not None
            else:
                # ── Ask user if they want to track this position
                position_id = ask_track_position(scanner_result, tracker, confidence)
                _just_tracked = position_id is not None

            if _just_tracked:
                action_state.update(key, "HOLD", confidence)
                logger.info(f"  {ticker} now tracked — action state set to HOLD")

        # Only update action state if we didn't already handle it above
        # If we notified (should_notify=True), state was already set inside the block
        # If exploration tick, don't update state — real signal may fire next tick
        if not should_notify and not is_explore_tick:
            action_state.update(key, action, confidence)

        rec_id = db.insert_recommendation({
            "timestamp":       snapshot["timestamp"],
            "ticker":          ticker,
            "strategy":        trade.get("strategy"),
            "direction":       trade.get("direction"),
            "strike":          main_leg.get("strike"),
            "expiration":      main_leg.get("exp") or trade.get("exp"),
            "action":          action,
            "confidence":      confidence,
            "reasons":         _json_dumps(reasons[:5]),
            "market_snapshot": _json_dumps(snapshot),
            "notified_user":   1 if should_notify else 0,
            "position_id":     position_id
        })

        if candidate_eval_id:
            try:
                with db.get_connection() as conn:
                    conn.execute(
                        """UPDATE candidate_evaluations
                           SET recommendation_id = ?, position_id = ?
                           WHERE id = ?""",
                        (rec_id, position_id, candidate_eval_id)
                    )
            except Exception as e:
                logger.warning(f"Candidate evaluation link failed for {ticker}: {e}")

        db.log_journal_event(
            "ENTRY_REC",
            ticker=ticker,
            action=action,
            confidence=confidence,
            reason_summary=f"{action} {ticker} conf={confidence:.2f}",
            details={
                "reasons":    reasons[:5],
                "entry":      pricing.get("entry"),
                "stop":       pricing.get("stop"),
                "target":     pricing.get("target"),
                "strategy":   trade.get("strategy"),
                "confluence": scanner_result.get("confluence", {}).get("score"),
                "rec_id":     rec_id,
                "candidate_eval_id": candidate_eval_id,
                "notified":   should_notify,
                "tracked":    position_id is not None
            }
        )

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


def _clear_price_cache(position_id: int):
    """Remove cached price for a closed position."""
    _price_cache.pop(position_id, None)


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


# =============================================================================
#  CLI COMMAND HANDLERS
# =============================================================================

def _cmd_close_position(tracker: PositionTracker, position_id: int):
    """
    Mark a single position as manually closed.
    Asks for exit price, records the close, sets cooldown.
    Use this after manually closing a trade in ThinkorSwim.
    """
    position = tracker.get_position(position_id)
    if not position:
        # Check if it exists but is already closed
        pos = db.get_position_by_id(position_id)
        if pos:
            print(f"  Position #{position_id} ({pos['ticker']}) is already closed.")
        else:
            print(f"  Position #{position_id} not found.")
        return

    ticker     = position.get("ticker", "?")
    entry      = position.get("entry_price", 0)
    entry_cost = position.get("entry_cost", 0)

    print(f"\n  Closing position #{position_id}: {ticker}")
    print(f"  Entry: ${entry}  |  Entry cost: ${entry_cost:.2f}")

    fill_str = input(f"  Exit price (what did you close at?): ").strip()
    try:
        exit_price = float(fill_str)
        if exit_price <= 0:
            print(f"  Invalid price (must be > 0) — cancelled")
            return
    except ValueError:
        print(f"  Invalid price — cancelled")
        return

    result = tracker.close_position(position_id, exit_price, "MANUAL")
    pnl = result.get("realized_pnl", 0)
    r   = result.get("realized_r", 0)
    sign = "+" if pnl >= 0 else ""
    print(f"  ✓ #{position_id} {ticker} closed @ ${exit_price}")
    print(f"    P&L: {sign}${pnl:.2f} ({r:+.2f}R)")
    print(f"    24hr cooldown set on {ticker}")


def _cmd_close_all_positions(tracker: PositionTracker):
    """
    Mark ALL open positions as manually closed.
    Asks for exit price per position.
    Use this after closing all trades in ThinkorSwim.
    """
    if tracker.open_count == 0:
        print("  No open positions to close.")
        return

    print(f"\n  Closing all {tracker.open_count} open position(s):")
    tracker.print_summary()

    confirm = input("\n  Close ALL positions? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    for position_id, position in list(tracker.open_positions.items()):
        ticker = position.get("ticker", "?")
        entry  = position.get("entry_price", 0)
        print(f"\n  Position #{position_id}: {ticker} (entry ${entry})")
        fill_str = input(f"  Exit price for {ticker} (or press enter to skip): ").strip()

        if not fill_str:
            print(f"  Skipping {ticker}")
            continue

        try:
            exit_price = float(fill_str)
            if exit_price <= 0:
                print(f"  Invalid price (must be > 0) — skipping {ticker}")
                continue
        except ValueError:
            print(f"  Invalid price — skipping {ticker}")
            continue

        result = tracker.close_position(position_id, exit_price, "MANUAL")
        pnl  = result.get("realized_pnl", 0)
        r    = result.get("realized_r", 0)
        sign = "+" if pnl >= 0 else ""
        print(f"  ✓ {ticker} closed @ ${exit_price} — P&L: {sign}${pnl:.2f} ({r:+.2f}R)")

    print(f"\n  Done. All closed positions recorded.")


def _cmd_delete_position(position_id: int):
    """
    Delete a position entirely from the database — no record kept.
    Use this for orders that never filled (working orders you cancelled).
    NOT for real closed trades — use --close for those.
    """
    pos = db.get_position_by_id(position_id)
    if not pos:
        print(f"  Position #{position_id} not found.")
        return

    ticker = pos.get("ticker", "?")
    status = pos.get("status", "?")

    print(f"\n  ⚠  About to permanently DELETE position #{position_id}: {ticker} ({status})")
    print(f"  Use --close instead if this was a real trade that filled.")
    confirm = input(f"  Delete #{position_id} {ticker}? (y/n): ").strip().lower()

    if confirm != "y":
        print("  Cancelled.")
        return

    with db.get_connection() as conn:
        conn.execute("DELETE FROM tick_snapshots WHERE position_id = ?", (position_id,))
        conn.execute("DELETE FROM recommendations WHERE position_id = ?", (position_id,))
        conn.execute("DELETE FROM positions WHERE id = ?", (position_id,))

    db.log_journal_event(
        "POSITION_CLOSED",
        ticker=ticker,
        position_id=position_id,
        action="DELETE",
        reason_summary=f"Position #{position_id} {ticker} deleted (unfilled order)"
    )

    print(f"  ✓ Position #{position_id} {ticker} deleted from database.")
    print(f"  (No cooldown set — this was an unfilled order)")


def _cmd_reset(keep_weights: bool = True):
    """
    Clear trading data from the database.
    keep_weights=True  → clears positions, snapshots, cooldowns, recommendations
                         but keeps agent weights (learning preserved)
    keep_weights=False → wipes everything including agent weights (full fresh start)
    """
    label = "soft reset (keeping agent weights)" if keep_weights else "FULL reset (wiping everything)"
    print(f"\n  About to perform {label}.")
    print(f"  This will clear:")
    print(f"    - All positions (open and closed)")
    print(f"    - All tick snapshots")
    print(f"    - All recommendations")
    print(f"    - All cooldowns")
    print(f"    - Trade journal")
    if not keep_weights:
        print(f"    - Agent weights (learning starts over)")
    print()

    confirm = input("  Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("  Cancelled.")
        return

    # Clear all in-memory caches so stale data doesn't affect new session
    global _price_cache, _oi_cache, _earnings_cache, _earnings_cache_time
    _price_cache = {}
    _oi_cache    = {}
    _earnings_cache = {}
    _earnings_cache_time = {}

    with db.get_connection() as conn:
        # Delete in dependency order — children before parents
        # to avoid foreign key constraint violations
        conn.execute("DELETE FROM tick_snapshots")
        conn.execute("DELETE FROM recommendations")
        conn.execute("DELETE FROM trade_journal")
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM cooldowns")
        conn.execute("DELETE FROM system_state")
        if not keep_weights:
            conn.execute("DELETE FROM agent_weights")

    print(f"  ✓ Reset complete.")
    if keep_weights:
        print(f"  Agent weights preserved — learning continues from where it left off.")
    else:
        print(f"  Full fresh start — agent weights cleared.")
    print(f"  Run --status to confirm.")


def print_status(tracker: PositionTracker, agent: DecisionAgent):
    """Print full system status including live P&L on open positions."""
    W = 68
    print("\n" + "═" * W)
    print("  OPTIONS SCANNER — SYSTEM STATUS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * W)

    # ── Open positions with live P&L
    # Use AUTO limit if more positions open than paper limit (handles --status run separately)
    import sys
    is_auto = '--auto' in sys.argv or tracker.open_count > cfg.MAX_CONCURRENT_POSITIONS
    pos_limit = cfg.AUTO_MAX_POSITIONS if is_auto else cfg.MAX_CONCURRENT_POSITIONS
    print(f"\n  Open positions ({tracker.open_count}/{pos_limit}):")
    if not tracker.open_positions:
        print("    No open positions.")
    else:
        print(f"  {'ID':<5} {'Ticker':<7} {'Strategy':<20} {'Entry':>7} "
              f"{'Stop':>7} {'Target':>7} {'DTE':>4} {'P&L':>10} {'R':>7} {'Since':<16}")
        print("  " + "─" * (W - 2))

        total_unrealized = 0.0
        for pid, pos in tracker.open_positions.items():
            ticker     = pos.get("ticker", "?")
            entry      = pos.get("entry_price", 0) or 0
            stop       = pos.get("stop_price", 0) or 0
            target     = pos.get("target_price", 0) or 0
            contracts  = pos.get("contracts", 1) or 1
            entry_time = pos.get("entry_time", "")[:16]
            exp        = pos.get("expiration", "?")

            # DTE
            dte = "?"
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                dte    = str((exp_dt - datetime.now()).days)
            except Exception:
                pass

            # Live P&L — fetch current price
            current_price = get_current_option_price(pos)
            if current_price and current_price != entry:
                pnl  = (current_price - entry) * 100 * contracts
                r    = pnl / (entry * 100 * contracts) if entry > 0 else 0
                pnl_str = f"${pnl:+.0f}"
                r_str   = f"{r:+.2f}R"
                total_unrealized += pnl
            else:
                pnl_str = "fetching..."
                r_str   = "─"

            # Color hint based on P&L
            if pnl_str.startswith("$+"):
                pnl_display = f"[+] {pnl_str}"
            elif pnl_str.startswith("$-"):
                pnl_display = f"[-] {pnl_str}"
            else:
                pnl_display = pnl_str

            print(f"  #{pid:<4} {ticker:<7} "
                  f"{(pos.get('strategy') or '?'):<20} "
                  f"${entry:>6.2f} "
                  f"${stop:>6.2f} "
                  f"${target:>6.2f} "
                  f"{dte:>4} "
                  f"{pnl_str:>10} "
                  f"{r_str:>7} "
                  f"{entry_time:<16}")

        print("  " + "─" * (W - 2))
        sign = "+" if total_unrealized >= 0 else ""
        print(f"  Total unrealized P&L: {sign}${total_unrealized:.2f}")

        # ── Capital deployed
        total_committed = sum(
            (p.get("entry_price", 0) or 0) * 100 * (p.get("contracts", 1) or 1)
            for p in tracker.open_positions.values()
        )
        pct_deployed = (total_committed / cfg.ACCOUNT_SIZE * 100) if cfg.ACCOUNT_SIZE > 0 else 0
        print(f"  Capital deployed:      ${total_committed:,.2f} ({pct_deployed:.1f}% of account)")

    # ── Bankroll summary (shown when running in bankroll mode)
    # Read from DB journal to find starting bankroll and compute current pool
    try:
        import json as _json
        br_start = None
        br_current = None
        start_event = db._conn().execute(
            "SELECT details FROM trade_journal WHERE event_type='SYSTEM_START' "
            "AND details LIKE '%bankroll%' ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if start_event:
            details = _json.loads(start_event[0] or "{}")
            br_start = details.get("bankroll_start")
            br_current = details.get("bankroll_current")
        if br_start:
            gain = (br_current or br_start) - br_start
            sign = "+" if gain >= 0 else ""
            print(f"\n  Bankroll: ${br_current:,.2f} (started ${br_start:,.2f}, {sign}${gain:,.2f})")
    except Exception:
        pass

    # ── Today's realized P&L
    daily_pnl = db.get_daily_pnl()
    sign = "+" if daily_pnl >= 0 else ""
    print(f"\n  Today's realized P&L:  {sign}${daily_pnl:.2f}")

    # ── Recent closed trades
    recent = db.get_closed_positions(limit=5)
    if recent:
        print(f"\n  Recent closed trades (last 5):")
        print(f"  {'Ticker':<8} {'Strategy':<20} {'Entry':>7} {'Exit':>7} "
              f"{'P&L':>9} {'R':>7} {'Reason':<15} {'Date':<16}")
        print("  " + "─" * (W - 2))
        for p in recent:
            ticker   = p.get("ticker", "?")
            strategy = (p.get("strategy") or "?")[:18]
            entry    = p.get("entry_price", 0) or 0
            exit_p   = p.get("exit_price", 0) or 0
            pnl      = p.get("realized_pnl", 0) or 0
            r        = p.get("realized_r", 0) or 0
            reason   = (p.get("exit_reason") or "?")[:13]
            date     = (p.get("exit_time") or "")[:16]
            sign     = "+" if pnl >= 0 else ""
            print(f"  {ticker:<8} {strategy:<20} "
                  f"${entry:>6.2f} ${exit_p:>6.2f} "
                  f"{sign}${abs(pnl):>7.2f} "
                  f"{r:>+7.2f}R "
                  f"{reason:<15} {date:<16}")

    # ── Overall performance
    perf = db.get_performance_summary()
    total = perf.get("total_trades", 0)
    if total:
        wr   = perf.get("winners", 0) / max(total, 1) * 100
        print(f"\n  All-time performance ({total} closed trades):")
        print(f"    Win rate:    {wr:.1f}%  "
              f"({'profitable' if wr >= 50 else 'needs work'})")
        print(f"    Avg R:       {perf.get('avg_r', 0):+.3f}R")
        print(f"    Total P&L:   ${perf.get('total_pnl', 0):+,.2f}")
        print(f"    Best trade:  {perf.get('best_r', 0):+.2f}R")
        print(f"    Worst trade: {perf.get('worst_r', 0):+.2f}R")
    else:
        print(f"\n  No closed trades yet.")

    # ── Cooldowns
    with db.get_connection() as conn:
        cds = conn.execute(
            "SELECT ticker, cooldown_until FROM cooldowns "
            "WHERE cooldown_until > ? ORDER BY cooldown_until",
            (datetime.now().isoformat(),)
        ).fetchall()
    if cds:
        print(f"\n  Active cooldowns:")
        for cd in cds:
            until = cd["cooldown_until"][:16]
            print(f"    {cd['ticker']:<8} until {until}")

    # ── Agent
    print(f"\n  Agent learning:")
    print(f"    Enter model: {agent.enter_model.n_updates} updates  "
          f"mean reward {agent.enter_model.mean_reward:+.3f}R")
    print(f"    Exit model:  {agent.exit_model.n_updates} updates  "
          f"mean reward {agent.exit_model.mean_reward:+.3f}R")
    if agent.enter_model.n_updates < 10:
        remaining = 10 - agent.enter_model.n_updates
        print(f"    ({remaining} more closed trades before learning activates)")

    print("\n" + "═" * W + "\n")


# =============================================================================
#  TICK STATUS BAR
# =============================================================================

def _print_tick_bar(tracker: PositionTracker, tick: int, regime: Dict):
    """
    Print a compact one-line status bar every tick.
    Shows time, tick count, regime, market hours status,
    and a summary of all open positions with live P&L.

    Example:
      [10:05:22] tick=12 RISK_OFF VIX=31  |  CRWD#1 +$124 +0.25R  |  MU#2 +$47 +0.09R
    """
    et_str = ""
    in_hours = is_market_hours_for_entry()
    try:
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        et_str = datetime.now(et).strftime("%H:%M ET")
    except Exception:
        et_str = datetime.now().strftime("%H:%M")

    regime_str = regime.get("regime", "?") if isinstance(regime, dict) else "?"
    vix_str    = f"VIX={regime.get('vix', '?')}" if isinstance(regime, dict) else ""
    hours_str  = "" if in_hours else " [OUTSIDE HOURS]"

    parts = [f"[{et_str}] tick={tick} {regime_str} {vix_str}{hours_str}"]

    if tracker.open_count == 0:
        parts.append("no open positions")
    else:
        for pid, pos in tracker.open_positions.items():
            ticker    = pos.get("ticker", "?")
            entry     = pos.get("entry_price", 0) or 0
            contracts = pos.get("contracts", 1) or 1
            cur       = get_current_option_price(pos)
            if cur and cur != entry and entry > 0:
                pnl = (cur - entry) * 100 * contracts
                r   = pnl / (entry * 100 * contracts)
                sign = "+" if pnl >= 0 else ""
                parts.append(f"{ticker}#{pid} {sign}${pnl:.0f} {r:+.2f}R")
            else:
                parts.append(f"{ticker}#{pid} fetching...")

    print("  " + "  |  ".join(parts))


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

def _prompt_and_store_keys():
    """
    Check for required API keys in DB. Prompt user if missing.
    Keys are stored in system_state under file-specific keys so
    run.py and run_live.py can have separate Alpaca accounts.
    """
    
    changed = False

    # ── Tradier (shared — same key for both instances)
    tradier_key = db.get_state("tradier_api_key", "")
    if not tradier_key:
        print("\n  Tradier API key not set.")
        tradier_key = input("  Enter Tradier API key: ").strip()
        if tradier_key:
            db.set_state("tradier_api_key", tradier_key)
            changed = True

    # ── Schwab (optional — only prompted if --broker schwab)
    import sys
    if "--broker" in sys.argv and "schwab" in sys.argv:
        schwab_client_id = db.get_state("schwab_client_id", "")
        if not schwab_client_id:
            print("\n  Schwab Client ID not set.")
            schwab_client_id = input("  Enter Schwab Client ID: ").strip()
            if schwab_client_id:
                db.set_state("schwab_client_id", schwab_client_id)
                changed = True

        schwab_client_secret = db.get_state("schwab_client_secret", "")
        if not schwab_client_secret:
            print("\n  Schwab Client Secret not set.")
            schwab_client_secret = input("  Enter Schwab Client Secret: ").strip()
            if schwab_client_secret:
                db.set_state("schwab_client_secret", schwab_client_secret)
                changed = True

        schwab_account = db.get_state("schwab_account_number", "")
        if not schwab_account:
            print("\n  Schwab Account Number not set.")
            schwab_account = input("  Enter Schwab Account Number: ").strip()
            if schwab_account:
                db.set_state("schwab_account_number", schwab_account)
                changed = True

    # ── Alpaca (mid account)
    alpaca_key = db.get_state("alpaca_api_key_live", "")
    if not alpaca_key:
        print("\n  Alpaca API key not set (mid account).")
        alpaca_key = input("  Enter Alpaca API key: ").strip()
        if alpaca_key:
            db.set_state("alpaca_api_key_live", alpaca_key)
            changed = True

    alpaca_secret = db.get_state("alpaca_secret_key_live", "")
    if not alpaca_secret:
        print("\n  Alpaca secret key not set (mid account).")
        alpaca_secret = input("  Enter Alpaca secret key: ").strip()
        if alpaca_secret:
            db.set_state("alpaca_secret_key_live", alpaca_secret)
            changed = True

    if changed:
        print("  Keys saved.\n")

    # ── Verify keys
    print("  Verifying API keys...")

    # Tradier check
    try:
        import requests as _r
        tr = _r.get(
            "https://api.tradier.com/v1/markets/clock",
            headers={"Authorization": f"Bearer {tradier_key}", "Accept": "application/json"},
            timeout=6
        )
        if tr.status_code == 200:
            print("  Tradier:  OK")
        else:
            print(f"  Tradier:  FAILED (HTTP {tr.status_code}) — check your key")
    except Exception as e:
        print(f"  Tradier:  FAILED ({e})")

    # Alpaca check
    try:
        import requests as _r
        ar = _r.get(
            "https://paper-api.alpaca.markets/v2/account",
            headers={"APCA-API-KEY-ID": alpaca_key, "APCA-API-SECRET-KEY": alpaca_secret},
            timeout=6
        )
        if ar.status_code == 200:
            acct = ar.json()
            cash = float(acct.get("cash", 0))
            print(f"  Alpaca (mid account): OK — cash ${cash:,.2f}")
        else:
            print(f"  Alpaca (mid account): FAILED (HTTP {ar.status_code}) — check your keys")
    except Exception as e:
        print(f"  Alpaca (mid account): FAILED ({e})")

    print()

    return tradier_key, alpaca_key, alpaca_secret


def main():
    global DEBUG_MODE

    parser = argparse.ArgumentParser(description="Options Scanner RL Loop")
    parser.add_argument("--paper",   action="store_true", help="Paper trading mode")
    parser.add_argument("--debug",   action="store_true", help="Debug/verbose mode")
    parser.add_argument("--status",  action="store_true", help="Show status and exit")
    parser.add_argument("--weights", action="store_true", help="Show agent weights and exit")
    parser.add_argument("--close",   type=int, metavar="ID",
                        help="Mark position ID as manually closed and remove it")
    parser.add_argument("--close-all", action="store_true",
                        help="Remove ALL open positions (use after manual close in ThinkorSwim)")
    parser.add_argument("--delete",  type=int, metavar="ID",
                        help="Delete a position entirely (no record kept — use for unfilled orders)")
    parser.add_argument("--reset",   action="store_true",
                        help="Clear all positions, snapshots, cooldowns and recommendations (keeps agent weights)")
    parser.add_argument("--reset-all", action="store_true",
                        help="Wipe entire database including agent weights — full fresh start")
    parser.add_argument("--auto",    action="store_true",
                        help="Fully autonomous mode — no user input, auto enters/exits based on signals. "
                             "No position limit. Use for unattended proof-of-concept runs.")
    parser.add_argument("--bankroll", type=float, default=None, metavar="AMOUNT",
                        help="Bankroll mode — specify a starting dollar amount (e.g. --bankroll 5000). "
                             "System trades autonomously using only that pool. Profits compound back in. "
                             "1 contract per trade. Stops entering when bankroll can't cover next premium.")
    parser.add_argument("--broker", type=str, default="alpaca",
                        choices=["alpaca", "schwab"],
                        help="Broker to use for order execution (default: alpaca)")
    parser.add_argument("--live-paper", action="store_true",
                        help="Live paper mode — sends real orders to Alpaca paper account. "
                             "Trades visible at app.alpaca.markets/paper-trading/overview. "
                             "Use with --bankroll or --auto. Requires ALPACA_API_KEY and ALPACA_SECRET_KEY.")
    parser.add_argument("--reset-keys", action="store_true",
                        help="Clear stored API keys and re-prompt on next run.")
    args = parser.parse_args()

    if args.debug:
        cfg.DEBUG_MODE = True
        logging.getLogger().setLevel(logging.DEBUG)

    DEBUG_MODE      = cfg.DEBUG_MODE
    AUTO_MODE       = args.auto or (args.bankroll is not None) or args.live_paper
    BANKROLL_MODE   = args.bankroll is not None
    BANKROLL        = [args.bankroll or 0.0]   # mutable list so inner functions can update it
    LIVE_PAPER_MODE = args.live_paper

    # ── Initialize
    db.initialize_database()

    # ── One-shot pre-init commands (no keys needed)
    if args.reset:
        db.initialize_database()
        _cmd_reset(keep_weights=True)
        return

    if args.reset_all:
        db.initialize_database()
        _cmd_reset(keep_weights=False)
        return

    # ── Swap broker module based on --broker flag
    global BROKER_AVAILABLE, broker
    if args.broker == "schwab":
        try:
            import broker_schwab as broker
            # Inject Schwab keys from DB
            broker.CLIENT_ID      = db.get_state("schwab_client_id", "")
            broker.CLIENT_SECRET  = db.get_state("schwab_client_secret", "")
            broker.ACCOUNT_NUMBER = db.get_state("schwab_account_number", "")
            BROKER_AVAILABLE = True
            logger.info("Using Schwab broker")
        except ImportError as e:
            logger.error(f"Could not import broker_schwab: {e}")
            BROKER_AVAILABLE = False
    else:
        logger.info("Using Alpaca broker")
        # BROKER_AVAILABLE already set at module level from initial import

    # ── Handle --reset-keys
    if args.reset_keys:
        db.set_state("tradier_api_key", "")
        db.set_state("alpaca_api_key_live", "")
        db.set_state("alpaca_secret_key_live", "")
        print("  API keys cleared — you will be prompted on next run.")
        return

    # ── Ensure API keys are set
    _tradier_key, _alpaca_key, _alpaca_secret = _prompt_and_store_keys()

    # ── Inject keys into modules
    if BROKER_AVAILABLE:
        broker.ALPACA_API_KEY    = _alpaca_key
        broker.ALPACA_SECRET_KEY = _alpaca_secret
    import options_scanner as _scanner_mod
    _scanner_mod.TRADIER_API_KEY = _tradier_key

    tracker      = PositionTracker()
    agent        = DecisionAgent()
    action_state = ActionStateTracker()

    # ── Restore or initialize bankroll
    if BANKROLL_MODE:
        stored = db.get_state("bankroll_remaining")
        if stored is not None:
            # Restore remaining balance from previous session
            BANKROLL[0] = float(stored)
            logger.info(f"Bankroll restored from DB: ${BANKROLL[0]:,.2f} remaining (started at ${args.bankroll or 0:,.2f})")
        else:
            # First run — save initial bankroll to DB
            db.set_state("bankroll_remaining", BANKROLL[0])
            logger.info(f"Bankroll initialized: ${BANKROLL[0]:,.2f}")

    # ── One-shot commands
    if args.status:
        print_status(tracker, agent)
        return

    if args.weights:
        agent.print_weight_summary()
        return

    if args.close:
        _cmd_close_position(tracker, args.close)
        return

    if args.close_all:
        _cmd_close_all_positions(tracker)
        return

    if args.delete:
        _cmd_delete_position(args.delete)
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
            "bankroll_start":      args.bankroll,
            "bankroll_current":    args.bankroll,
        }
    )

    in_hours = is_market_hours_for_entry()
    notify_info(
        f"RL loop started — "
        f"{'BANKROLL $' + str(int(args.bankroll)) if BANKROLL_MODE else 'AUTO' if args.auto else 'PAPER' if args.paper else 'LIVE'} mode, "
        f"{tracker.open_count} positions restored, "
        f"{'within' if in_hours else 'OUTSIDE'} market hours"
    )
    if args.auto:
        print("\n" + "█" * 68)
        print("  ██  AUTO MODE — no user input required  ██")
        print("  Positions will open and close automatically based on signals.")
        print("  Position limit removed. Monitor via --status in another terminal.")
        print("█" * 68 + "\n")

    if BANKROLL_MODE:
        print("\n" + "█" * 68)
        print(f"  ██  BANKROLL MODE — starting pool: ${args.bankroll:,.2f}  ██")
        print("  System trades autonomously from this fixed pool.")
        print("  Profits compound back in. 1 contract per trade.")
        print(f"  Entries stop when remaining bankroll < next premium cost.")
        print("█" * 68 + "\n")

    if LIVE_PAPER_MODE:
        print("\n" + "█" * 68)
        print("  ██  LIVE PAPER MODE — orders sent to Alpaca paper account  ██")
        print("  Tradier prod key: real-time market data")
        print("  Alpaca paper key: order execution + web dashboard")
        print("  Dashboard: https://app.alpaca.markets/paper-trading/overview")
        if not BROKER_AVAILABLE:
            print("  ✗ ERROR: broker.py not found — cannot execute orders")
        elif not broker._is_configured():
            missing = []
            if not broker.ALPACA_API_KEY:    missing.append("ALPACA_API_KEY")
            if not broker.ALPACA_SECRET_KEY: missing.append("ALPACA_SECRET_KEY")
            print(f"  ✗ ERROR: Missing environment variables: {', '.join(missing)}")
            print("  Set them and restart.")
        else:
            balance = broker.get_account_balance()
            if balance is not None:
                print(f"  ✓ Sandbox connected — account balance: ${balance:,.2f}")
            else:
                print("  ⚠ Sandbox connected but balance unavailable")
        print("█" * 68 + "\n")

    print_status(tracker, agent)

    # ── Main loop state
    last_scanner_run   = 0.0
    last_scanner_results: List[Dict] = []
    regime: Dict = {}
    tick = 0
    _pending_orders: Dict[str, Dict] = load_pending_orders()  # restored from DB on startup
    _pending_exit_orders: Dict[str, Dict] = load_pending_exit_orders()
    if _pending_orders:
        logger.info(f"Restored {len(_pending_orders)} pending order(s) from previous session")
    if _pending_exit_orders:
        logger.info(f"Restored {len(_pending_exit_orders)} pending exit order(s) from previous session")

    logger.info(
        f"Starting 60-second loop "
        f"({'PAPER' if args.paper else 'LIVE'} mode)"
    )


    # ── Startup Alpaca sync — catch any fills that happened while offline
    if LIVE_PAPER_MODE and BROKER_AVAILABLE and broker._is_configured():
        try:
            logger.info("Syncing open positions from Alpaca on startup...")
            alpaca_positions = broker.get_positions()
            alpaca_symbols = {p["symbol"] for p in alpaca_positions}

            # Get tracked symbols from DB — include both long and short legs of spreads
            tracked_symbols = set()
            for pos in tracker.open_positions.values():
                sym = broker.build_option_symbol(
                    pos.get("ticker",""), pos.get("expiration",""),
                    pos.get("option_type","call"), pos.get("strike",0)
                )
                tracked_symbols.add(sym)
                # Also add short leg of spreads if present
                try:
                    notes = json.loads(pos.get("notes") or "{}")
                    short_strike = notes.get("short_strike")
                    short_type = notes.get("short_option_type", pos.get("option_type","call"))
                    if short_strike:
                        short_sym = broker.build_option_symbol(
                            pos.get("ticker",""), pos.get("expiration",""),
                            short_type, short_strike
                        )
                        tracked_symbols.add(short_sym)
                except Exception:
                    pass

            # Find positions in Alpaca not in our DB
            untracked = [p for p in alpaca_positions if p["symbol"] not in tracked_symbols]
            if untracked:
                logger.warning(f"Found {len(untracked)} Alpaca position(s) not in DB — checking pending orders")
                for ap in untracked:
                    sym = ap["symbol"]
                    # Check if we have a pending order for this symbol
                    matched_oid = None
                    matched_data = None
                    for oid, pdata in list(_pending_orders.items()):
                        expected_sym = broker.build_option_symbol(
                            pdata.get("ticker",""), pdata.get("expiration",""),
                            pdata.get("option_type","call"), pdata.get("strike",0)
                        )
                        if expected_sym == sym:
                            matched_oid = oid
                            matched_data = pdata
                            break

                    if matched_data:
                        fill_price = float(ap.get("avg_entry_price", 0) or 0)
                        if fill_price > 0:
                            matched_data["entry_price"]  = fill_price
                            matched_data["entry_cost"]   = fill_price * 100 * matched_data.get("contracts", 1)
                            matched_data["stop_price"]   = round(fill_price * (1 - cfg.STOP_LOSS_PCT), 2)
                            matched_data["target_price"] = round(fill_price * (1 + cfg.PROFIT_TARGET_PCT), 2)
                            pid = db.insert_position(matched_data)
                            if pid:
                                candidate_eval_id = matched_data.get("candidate_evaluation_id")
                                if candidate_eval_id:
                                    try:
                                        with db.get_connection() as conn:
                                            conn.execute(
                                                "UPDATE candidate_evaluations SET position_id = ? WHERE id = ?",
                                                (pid, candidate_eval_id)
                                            )
                                    except Exception as e:
                                        logger.warning(
                                            f"Startup candidate evaluation link failed for {matched_data.get('ticker')}: {e}"
                                        )
                                _update_broker_order(
                                    matched_oid,
                                    "FILLED",
                                    filled_price=fill_price,
                                    position_id=pid,
                                    raw_payload=ap,
                                    notes="startup_sync_fill",
                                )
                                loaded = db.get_position_by_id(pid)
                                if loaded:
                                    tracker._open[pid] = loaded
                                print(f"  [STARTUP SYNC] Recovered {matched_data.get('ticker')} #{pid} @ ${fill_price:.2f} from Alpaca")
                                logger.info(f"  Startup sync: recovered {sym} #{pid}")
                            if matched_oid:
                                _pending_orders.pop(matched_oid, None)
                    else:
                        logger.warning(f"  Alpaca has {sym} but no pending order match — position may be from outside this system")

                save_pending_orders(_pending_orders)
            else:
                logger.info("Startup sync complete — all Alpaca positions accounted for")
        except Exception as e:
            logger.warning(f"Startup Alpaca sync failed: {e}")

    while _running:
        tick_start = time.time()
        tick += 1

        try:
            # ── Scanner refresh (market hours only)
            time_since_scan = tick_start - last_scanner_run
            in_hours_for_scan = is_market_hours_for_entry()
            if (time_since_scan >= cfg.SCANNER_RUN_INTERVAL or not last_scanner_results) and in_hours_for_scan:
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
            elif not in_hours_for_scan and time_since_scan >= cfg.SCANNER_RUN_INTERVAL:
                logger.debug(f"Tick {tick}: Outside market hours — skipping scanner refresh")

            # ── Evaluate open positions
            if tracker.open_count > 0:
                evaluate_open_positions(
                    tracker, agent, action_state, last_scanner_results, regime,
                    auto_mode=AUTO_MODE,
                    bankroll=BANKROLL if BANKROLL_MODE else None,
                    live_paper=LIVE_PAPER_MODE,
                    pending_exit_orders=_pending_exit_orders
                )

            # ── Reconcile pending exit orders before considering new entries
            if LIVE_PAPER_MODE and BROKER_AVAILABLE and _pending_exit_orders:
                try:
                    import requests as _req
                    resolved_exit_oids = []
                    for oid, exit_data in list(_pending_exit_orders.items()):
                        r = _req.get(
                            f"{broker.ALPACA_BASE}/orders/{oid}",
                            headers=broker._headers(), timeout=8
                        )
                        if r.status_code != 200:
                            continue

                        order = r.json()
                        status = order.get("status", "").lower()
                        position_id = int(exit_data.get("position_id", 0) or 0)
                        ticker_r = exit_data.get("ticker", "?")

                        if status == "filled":
                            fill_price = float(order.get("filled_avg_price", 0) or 0)
                            if fill_price <= 0:
                                fill_price = float(exit_data.get("estimated_exit_price", 0) or 0)
                            position = tracker.get_position(position_id) or exit_data.get("position", {})
                            if position and fill_price > 0:
                                _update_broker_order(
                                    oid,
                                    "FILLED",
                                    filled_price=fill_price,
                                    position_id=position_id,
                                    raw_payload=order,
                                    notes="exit_fill_confirmed",
                                )
                                result = tracker.close_position(
                                    position_id,
                                    fill_price,
                                    exit_data.get("exit_reason", "BROKER_EXIT")
                                )
                                realized_r = result.get("realized_r", 0)
                                if BANKROLL_MODE:
                                    proceeds = fill_price * 100 * position.get("contracts", 1)
                                    BANKROLL[0] = round(BANKROLL[0] + proceeds, 2)
                                    db.set_state("bankroll_remaining", BANKROLL[0])

                                entry_snap = json.loads(position.get("raw_scanner_data") or "{}")
                                entry_snap["regime_data"] = exit_data.get("regime", regime)
                                exit_snapshot = exit_data.get("snapshot") or {}
                                exit_snapshot["option_mid"] = fill_price
                                agent.update_on_close(
                                    position=position,
                                    exit_reason=exit_data.get("exit_reason", "BROKER_EXIT"),
                                    realized_r=realized_r,
                                    entry_market_snapshot=build_market_snapshot(entry_snap),
                                    exit_market_snapshot=exit_snapshot,
                                    ticks_held=_ticks_held(position),
                                    rolling_drawdown=_rolling_drawdown()
                                )
                                _clear_price_cache(position_id)
                                action_state.update(str(position_id), "EXIT", 1.0)
                                print(f"  [RECONCILE] Exit fill confirmed — CLOSED {ticker_r} #{position_id} @ ${fill_price:.2f} order={oid}")
                            resolved_exit_oids.append(oid)
                        elif status in ("canceled", "cancelled", "rejected", "expired"):
                            mapped_status = {
                                "canceled": "CANCELLED",
                                "cancelled": "CANCELLED",
                                "rejected": "REJECTED",
                                "expired": "EXPIRED",
                            }.get(status, status.upper())
                            _update_broker_order(
                                oid,
                                mapped_status,
                                position_id=position_id,
                                raw_payload=order,
                                notes="exit_not_filled",
                            )
                            logger.warning(
                                f"  Pending exit order {oid} for {ticker_r} {status}; "
                                "position remains open"
                            )
                            db.log_journal_event(
                                "EXIT_ORDER_FAILED",
                                ticker=ticker_r,
                                position_id=position_id,
                                reason_summary=f"Exit order {status}; position remains open",
                                details={"order_id": oid, "status": status}
                            )
                            action_state.update(str(position_id), "HOLD", 0.0)
                            resolved_exit_oids.append(oid)

                    for oid in resolved_exit_oids:
                        _pending_exit_orders.pop(oid, None)
                    if resolved_exit_oids:
                        save_pending_exit_orders(_pending_exit_orders)
                except Exception as e:
                    logger.warning(f"Exit reconciliation error: {e}")

            # ── Reconcile pending orders (check for late fills at Alpaca)
            if LIVE_PAPER_MODE and BROKER_AVAILABLE and _pending_orders:
                try:
                    filled_oids = []
                    for oid, pos_data in list(_pending_orders.items()):
                        import requests as _req

                        # ── Staleness check: cancel if order too old or price drifted too far
                        queued_at = pos_data.get("entry_time", "")
                        try:
                            from datetime import datetime as _dtc
                            queued_dt = _dtc.fromisoformat(queued_at)
                            age_hours = (_dtc.now() - queued_dt).total_seconds() / 3600
                            # Cancel if older than market close (6.5 hours from open)
                            if age_hours > 6.5:
                                broker.cancel_order(oid)
                                _update_broker_order(oid, "CANCELLED", notes="entry_order_age_cancelled")
                                logger.info(f"  [RECONCILE] Cancelled stale order {oid} ({pos_data.get('ticker')} aged {age_hours:.1f}h)")
                                ticker_r = pos_data.get("ticker", "")
                                if ticker_r:
                                    from datetime import timedelta
                                    cooldown_until = (datetime.now() + timedelta(hours=4)).isoformat()
                                    db.set_cooldown(ticker_r, cooldown_until, reason="order_age_cancelled")
                                filled_oids.append(oid)
                                continue
                        except Exception:
                            pass

                        # ── Price drift check: cancel if live price > 20% above actual limit
                        # Only run during market hours — options don't trade after hours
                        try:
                            ticker_r = pos_data.get("ticker", "")
                            occ_sym = broker.build_option_symbol(
                                ticker_r, pos_data.get("expiration",""),
                                pos_data.get("option_type","call"), pos_data.get("strike",0)
                            )
                            if get_live_option_quote and occ_sym and is_market_hours_for_entry():
                                live_q = get_live_option_quote(occ_sym)
                                live_ask = live_q.get("ask", 0)
                                limit_price = pos_data.get("limit_price", pos_data.get("entry_price", 0))
                                if live_ask > 0 and limit_price > 0 and live_ask > limit_price * 1.20:
                                    broker.cancel_order(oid)
                                    _update_broker_order(oid, "CANCELLED", notes="entry_order_drift_cancelled")
                                    logger.info(f"  [RECONCILE] Cancelled {ticker_r} order {oid} — price drifted too far (limit=${limit_price:.2f} ask=${live_ask:.2f})")
                                    # Cooldown ticker so it doesn't re-enter this session
                                    from datetime import timedelta
                                    cooldown_until = (datetime.now() + timedelta(hours=4)).isoformat()
                                    db.set_cooldown(ticker_r, cooldown_until, reason="order_drift_cancelled")
                                    filled_oids.append(oid)
                                    continue
                        except Exception:
                            pass

                        r = _req.get(
                            f"{broker.ALPACA_BASE}/orders/{oid}",
                            headers=broker._headers(), timeout=8
                        )
                        if r.status_code == 200:
                            order = r.json()
                            status = order.get("status", "").lower()
                            if status == "filled":
                                fill_price = float(order.get("filled_avg_price", 0) or 0)
                                if fill_price > 0:
                                    fill_cost = fill_price * 100 * pos_data.get("contracts", 1)
                                    ticker_r  = pos_data.get("ticker", "?")

                                    pos_data["entry_price"]  = fill_price
                                    pos_data["entry_cost"]   = fill_cost
                                    pos_data["stop_price"]   = round(fill_price * (1 - cfg.STOP_LOSS_PCT), 2)
                                    pos_data["target_price"] = round(fill_price * (1 + cfg.PROFIT_TARGET_PCT), 2)
                                    from database import insert_position as _ins2
                                    pid = _ins2(pos_data)
                                    if pid:
                                        candidate_eval_id = pos_data.get("candidate_evaluation_id")
                                        if candidate_eval_id:
                                            try:
                                                with db.get_connection() as conn:
                                                    conn.execute(
                                                        "UPDATE candidate_evaluations SET position_id = ? WHERE id = ?",
                                                        (pid, candidate_eval_id)
                                                    )
                                            except Exception as e:
                                                logger.warning(
                                                    f"Candidate evaluation fill link failed for {ticker_r}: {e}"
                                                )
                                        _update_broker_order(
                                            oid,
                                            "FILLED",
                                            filled_price=fill_price,
                                            position_id=pid,
                                            raw_payload=order,
                                            notes="entry_fill_confirmed",
                                        )
                                        loaded = db.get_position_by_id(pid)
                                        if loaded:
                                            tracker._open[pid] = loaded
                                        print(f"  [RECONCILE] Late fill detected — ENTER {ticker_r} #{pid} @ ${fill_price:.2f} order={oid}")
                                        logger.info(f"  Reconciled late fill: {ticker_r} #{pid} order={oid}")
                                        if BANKROLL_MODE:
                                            BANKROLL[0] = round(BANKROLL[0] - fill_cost, 2)
                                            db.set_state("bankroll_remaining", BANKROLL[0])
                                            print(f"  [BANKROLL] Reconciled deduction — remaining=${BANKROLL[0]:,.2f}")
                                            # Cancel any remaining pending orders that would overdraw bankroll
                                            for other_oid, other_data in list(_pending_orders.items()):
                                                if other_oid == oid:
                                                    continue
                                                other_cost = other_data.get("entry_price", 0) * 100 * other_data.get("contracts", 1)
                                                if other_cost > BANKROLL[0]:
                                                    broker.cancel_order(other_oid)
                                                    _update_broker_order(other_oid, "CANCELLED", notes="bankroll_overdraw_cancelled")
                                                    filled_oids.append(other_oid)
                                                    print(f"  [BANKROLL] Cancelled pending {other_data.get('ticker')} order {other_oid} — would overdraw bankroll (need ${other_cost:.2f}, have ${BANKROLL[0]:.2f})")
                                filled_oids.append(oid)
                            elif status in ("canceled", "cancelled", "rejected", "expired"):
                                mapped_status = {
                                    "canceled": "CANCELLED",
                                    "cancelled": "CANCELLED",
                                    "rejected": "REJECTED",
                                    "expired": "EXPIRED",
                                }.get(status, status.upper())
                                _update_broker_order(
                                    oid,
                                    mapped_status,
                                    raw_payload=order,
                                    notes="entry_not_filled",
                                )
                                logger.info(f"  Pending order {oid} {status} — removing from reconciliation queue")
                                filled_oids.append(oid)
                    for oid in filled_oids:
                        _pending_orders.pop(oid, None)
                    if filled_oids:
                        save_pending_orders(_pending_orders)
                except Exception as e:
                    logger.warning(f"Reconciliation error: {e}")

            # ── Evaluate new entry candidates
            evaluate_new_candidates(
                tracker, agent, action_state, last_scanner_results, regime,
                auto_mode=AUTO_MODE,
                bankroll=BANKROLL if BANKROLL_MODE else None,
                live_paper=LIVE_PAPER_MODE,
                pending_orders=_pending_orders
            )

            # ── Print compact tick status bar
            # Suppress overnight when outside hours and no positions — avoids spam
            if is_market_hours_for_entry() or tracker.open_count > 0:
                _print_tick_bar(tracker, tick, regime)

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
