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

        # Get current option price from scanner or last known
        current_option_price = (
            scanner_result.get("trade", {}).get("main_leg", {}).get("mid")
            or position.get("entry_price")
        )

        # Build snapshot
        snapshot = build_market_snapshot(scanner_result, position, current_option_price)

        # ── HARD RULES FIRST — non-negotiable
        must_exit, hard_reason = tracker.check_hard_exit_rules(position, current_option_price)

        if must_exit:
            # Execute hard exit
            result = tracker.close_position(position_id, current_option_price, hard_reason)
            realized_r = result.get("realized_r", 0)

            # Notify with high priority regardless of confidence threshold
            if "STOP" in hard_reason.upper():
                notify_stop_hit(
                    ticker=ticker,
                    loss=result.get("realized_pnl", 0),
                    position_id=position_id
                )
            else:
                notify_exit(
                    ticker=ticker,
                    confidence=1.0,
                    reasons=[f"Hard rule: {hard_reason}"],
                    unrealized_pnl=result.get("realized_pnl"),
                    force=True
                )

            # ── Print copy-paste close command for hard exits too
            close_cmd = build_close_command(position_id, ticker)
            sep = "=" * 64
            print("\n" + sep)
            print("  !! CLOSE THIS POSITION IN THINKORSWIM NOW !!")
            print("  OPEN TERMINAL 2 (Cmd+T), then paste this command:")
            print(sep)
            print(close_cmd)
            print("  (Replace FILL_PRICE with your actual exit fill price)")
            print(sep + "\n")

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
            "feature_vector": json.dumps([]),   # populated in agent internally
            "agent_action":   action,
            "agent_confidence": confidence,
            "agent_reasons":  json.dumps(reasons[:5])
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

            # ── Print copy-paste close command
            close_cmd = build_close_command(position_id, ticker)
            sep = "=" * 64
            print("\n" + sep)
            print("  OPEN TERMINAL 2 (Cmd+T), then paste this command:")
            print(sep)
            print(close_cmd)
            print("  (Replace FILL_PRICE with your actual exit fill price)")
            print(sep + "\n")

            db.insert_recommendation({
                "timestamp":      snapshot["timestamp"],
                "ticker":         ticker,
                "strategy":       position.get("strategy"),
                "direction":      position.get("direction"),
                "strike":         position.get("strike"),
                "expiration":     position.get("expiration"),
                "action":         action,
                "confidence":     confidence,
                "reasons":        json.dumps(reasons[:5]),
                "market_snapshot": json.dumps(snapshot, default=str),
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
            pricing = scanner_result.get("pricing", {})
            trade   = scanner_result.get("trade", {})
            notify_entry(
                ticker=ticker,
                confidence=confidence,
                reasons=reasons[:5],
                entry=pricing.get("entry"),
                stop=pricing.get("stop"),
                target=pricing.get("target"),
                strategy=trade.get("strategy")
            )

            # ── Print copy-paste tracker command so user never has to remember it
            track_cmd = build_track_command(scanner_result)
            sep = "=" * 64
            print("\n" + sep)
            print("  OPEN TERMINAL 2 (Cmd+T), then paste this command:")
            print(sep)
            print(track_cmd)
            print("  (Replace FILL_PRICE with your actual fill e.g. 2.80)")
            print(sep + "\n")

            rec_id = db.insert_recommendation({
                "timestamp":       snapshot["timestamp"],
                "ticker":          ticker,
                "strategy":        trade.get("strategy"),
                "direction":       trade.get("direction"),
                "strike":          trade.get("main_leg", {}).get("strike"),
                "expiration":      trade.get("exp"),
                "action":          action,
                "confidence":      confidence,
                "reasons":         json.dumps(reasons[:5]),
                "market_snapshot": json.dumps(snapshot, default=str),
                "notified_user":   1,
                "position_id":     None
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
                    "rec_id":     rec_id
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
    """Compute recent rolling drawdown from closed positions."""
    recent = db.get_closed_positions(limit=10)
    if not recent:
        return 0.0
    total_pnl = sum(p.get("realized_pnl", 0) for p in recent)
    return total_pnl / cfg.ACCOUNT_SIZE


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
