"""
=============================================================================
position_tracker.py — Position State Management
=============================================================================
Tracks recommended and open positions, enforces hard risk rules,
manages cooldowns, and maintains the position lifecycle from
recommendation → open → closed.

Hard risk rules enforced here:
  - Max concurrent positions
  - Stop loss trigger
  - DTE force-close
  - Cooldown / no-chase
  - Daily drawdown block
=============================================================================
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import database as db
from config import (
    MAX_CONCURRENT_POSITIONS,
    STOP_LOSS_PCT,
    PROFIT_TARGET_PCT,
    CLOSE_BEFORE_DTE,
    COOLDOWN_HOURS,
    NO_REENTRY_WHILE_OPEN,
    MAX_DAILY_DRAWDOWN_PCT,
    ACCOUNT_SIZE,
    DEBUG_MODE,
)

logger = logging.getLogger(__name__)


# =============================================================================
#  POSITION LIFECYCLE
# =============================================================================

class PositionTracker:
    """
    Manages all open and closed positions.
    Acts as the single source of truth for position state.
    Hard risk controls live here — they cannot be overridden by the agent.
    """

    def __init__(self):
        self._open: Dict[int, Dict] = {}    # position_id -> position dict
        self._reload_from_db()
        logger.info(f"PositionTracker initialized — {len(self._open)} open positions restored")

    # ─── Initialization ──────────────────────────────────────────────────────

    def _reload_from_db(self):
        """Restore open positions from database on startup."""
        positions = db.get_open_positions()
        self._open = {p["id"]: p for p in positions}
        if self._open:
            logger.info(f"Restored {len(self._open)} open positions from DB")
            for pid, pos in self._open.items():
                logger.info(
                    f"  #{pid} {pos['ticker']} {pos['strategy']} "
                    f"entry=${pos['entry_price']} entered={pos['entry_time'][:16]}"
                )

    # ─── Hard Risk Checks ────────────────────────────────────────────────────

    def can_enter(self, ticker: str) -> Tuple[bool, str]:
        """
        Check all hard risk rules before allowing a new entry.
        Returns (allowed: bool, reason: str)
        These checks CANNOT be overridden by the learning layer.
        """
        # 1. Max concurrent positions
        if len(self._open) >= MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions reached ({MAX_CONCURRENT_POSITIONS})"

        # 2. No re-entry while position already open on this ticker
        if NO_REENTRY_WHILE_OPEN:
            for pos in self._open.values():
                if pos["ticker"].upper() == ticker.upper():
                    return False, f"Position already open on {ticker}"

        # 3. Cooldown check
        if db.is_on_cooldown(ticker):
            cd = db.get_cooldown(ticker)
            until = cd["cooldown_until"][:16] if cd else "unknown"
            return False, f"{ticker} on cooldown until {until}"

        # 4. Daily drawdown block
        daily_pnl = db.get_daily_pnl()
        max_daily_loss = ACCOUNT_SIZE * MAX_DAILY_DRAWDOWN_PCT
        if daily_pnl < -max_daily_loss:
            return False, (
                f"Daily drawdown limit hit: ${daily_pnl:.2f} loss "
                f"(limit: -${max_daily_loss:.0f})"
            )

        return True, "OK"

    def check_hard_exit_rules(self, position: Dict, current_option_price: float
                               ) -> Tuple[bool, str]:
        """
        Check if a position must be force-closed by a hard rule.
        Returns (must_exit: bool, reason: str)
        """
        # ── Grace period: never fire hard rules within 2 minutes of opening
        # This prevents a bad current_price reading on the first tick from
        # immediately triggering a stop right after the user says y to track.
        entry_time = position.get("entry_time", "")
        if entry_time:
            try:
                entry_dt   = datetime.fromisoformat(entry_time)
                secs_held  = (datetime.now() - entry_dt).total_seconds()
                if secs_held < 120:
                    return False, ""   # too early — give it 2 minutes
            except Exception:
                pass

        entry_price = position["entry_price"]
        entry_cost  = position["entry_cost"]
        contracts   = position.get("contracts", 1)

        # Current value
        current_value = current_option_price * 100 * contracts
        pnl           = current_value - entry_cost
        pct_of_entry  = current_option_price / entry_price if entry_price > 0 else 1

        # 1. Stop loss
        if pct_of_entry <= (1 - STOP_LOSS_PCT):
            loss = entry_cost - current_value
            return True, f"STOP_LOSS: down {(1-pct_of_entry)*100:.0f}% (${loss:.2f})"

        # 2. Profit target
        if pct_of_entry >= (1 + PROFIT_TARGET_PCT):
            gain = current_value - entry_cost
            return True, f"TARGET_HIT: up {(pct_of_entry-1)*100:.0f}% (+${gain:.2f})"

        # 3. DTE force-close
        if position.get("expiration"):
            try:
                exp_dt = datetime.strptime(position["expiration"], "%Y-%m-%d")
                dte    = (exp_dt - datetime.now()).days
                if dte <= CLOSE_BEFORE_DTE:
                    return True, f"DTE_EXPIRY: only {dte} days to expiry (threshold: {CLOSE_BEFORE_DTE})"
            except Exception:
                pass

        return False, ""

    # ─── Position CRUD ───────────────────────────────────────────────────────

    def open_position(self, scanner_result: Dict, recommendation_id: int = None
                       ) -> Optional[int]:
        """
        Record a new open position from a scanner recommendation.
        Returns position_id or None if blocked by hard rules.
        """
        ticker = scanner_result.get("ticker", "?")

        # Hard rule check
        allowed, reason = self.can_enter(ticker)
        if not allowed:
            logger.info(f"Entry blocked for {ticker}: {reason}")
            db.log_journal_event(
                "RISK_BLOCK", ticker=ticker,
                reason_summary=f"Entry blocked: {reason}",
                details={"reason": reason}
            )
            return None

        trade   = scanner_result.get("trade", {})
        pricing = scanner_result.get("pricing", {})
        vol     = scanner_result.get("vol", {})
        main    = trade.get("main_leg", {})

        entry_price = pricing.get("entry", main.get("mid", 0))
        stop_price  = pricing.get("stop", 0)
        target_price = pricing.get("target", 0)
        contracts   = pricing.get("contracts", 1)
        entry_cost  = entry_price * 100 * contracts

        # Compute DTE at entry
        expiration = main.get("exp", trade.get("exp", ""))
        entry_dte  = None
        if expiration:
            try:
                exp_dt    = datetime.strptime(expiration, "%Y-%m-%d")
                entry_dte = (exp_dt - datetime.now()).days
            except Exception:
                pass

        data = {
            "ticker":           ticker,
            "strategy":         trade.get("strategy"),
            "direction":        trade.get("direction"),
            "option_type":      main.get("option_type"),
            "strike":           main.get("strike"),
            "expiration":       expiration,
            "entry_price":      entry_price,
            "entry_cost":       entry_cost,
            "contracts":        contracts,
            "stop_price":       stop_price,
            "target_price":     target_price,
            "entry_time":       datetime.now().isoformat(),
            "confluence_score": scanner_result.get("confluence", {}).get("score"),
            "entry_ivr":        vol.get("ivr"),
            "entry_dte":        entry_dte,
            "notes":            "",
            "raw_scanner_data": json.dumps(scanner_result, default=str)
        }

        position_id = db.insert_position(data)
        self._open[position_id] = db.get_position_by_id(position_id)

        db.log_journal_event(
            "POSITION_OPENED",
            ticker=ticker,
            position_id=position_id,
            action="ENTER",
            confidence=None,
            reason_summary=f"Opened {trade.get('strategy')} on {ticker} @ ${entry_price}",
            details={
                "entry_price": entry_price,
                "stop_price":  stop_price,
                "target_price": target_price,
                "contracts":   contracts,
                "entry_dte":   entry_dte,
                "recommendation_id": recommendation_id
            }
        )

        logger.info(
            f"Position opened: #{position_id} {ticker} "
            f"{trade.get('strategy')} entry=${entry_price} "
            f"stop=${stop_price} target=${target_price}"
        )
        return position_id

    def close_position(self, position_id: int, current_option_price: float,
                        exit_reason: str) -> Dict:
        """
        Close an open position and record outcome.
        Returns summary dict with realized P&L.
        """
        position = self._open.get(position_id)
        if not position:
            logger.warning(f"close_position called for unknown position {position_id}")
            return {}

        contracts   = position.get("contracts", 1)
        entry_cost  = position.get("entry_cost", 0)
        exit_value  = current_option_price * 100 * contracts
        realized_pnl = exit_value - entry_cost
        initial_risk = entry_cost
        realized_r   = realized_pnl / initial_risk if initial_risk > 0 else 0

        db.close_position(
            position_id  = position_id,
            exit_price   = current_option_price,
            exit_reason  = exit_reason,
            realized_pnl = realized_pnl,
            realized_r   = realized_r
        )

        # Set cooldown on this ticker
        cooldown_until = (
            datetime.now() + timedelta(hours=COOLDOWN_HOURS)
        ).isoformat()
        db.set_cooldown(
            ticker         = position["ticker"],
            cooldown_until = cooldown_until,
            reason         = f"Post-close cooldown after {exit_reason}"
        )

        db.log_journal_event(
            "POSITION_CLOSED",
            ticker      = position["ticker"],
            position_id = position_id,
            action      = "EXIT",
            confidence  = None,
            reason_summary = (
                f"Closed {position['ticker']} {exit_reason}: "
                f"P&L=${realized_pnl:+.2f} ({realized_r:+.2f}R)"
            ),
            details = {
                "exit_reason":   exit_reason,
                "exit_price":    current_option_price,
                "realized_pnl":  realized_pnl,
                "realized_r":    realized_r,
                "entry_price":   position.get("entry_price"),
                "entry_time":    position.get("entry_time")
            }
        )

        result = {
            "position_id":  position_id,
            "ticker":       position["ticker"],
            "exit_reason":  exit_reason,
            "exit_price":   current_option_price,
            "realized_pnl": realized_pnl,
            "realized_r":   realized_r
        }

        del self._open[position_id]
        logger.info(
            f"Position closed: #{position_id} {position['ticker']} "
            f"{exit_reason} P&L=${realized_pnl:+.2f} ({realized_r:+.2f}R)"
        )
        return result

    def manual_override_close(self, position_id: int, current_price: float,
                               reason: str = "MANUAL"):
        """Allow manual override close from CLI or external command."""
        return self.close_position(position_id, current_price, reason)

    def manual_override_open(self, ticker: str, strategy: str, strike: float,
                              expiration: str, entry_price: float,
                              contracts: int = 1) -> Optional[int]:
        """
        Allow manually entering a position that wasn't from the scanner.
        Used when you execute a trade the scanner recommended but want
        to track a slightly different fill price.
        """
        # Build a minimal scanner_result-like dict
        mock_result = {
            "ticker": ticker,
            "trade": {
                "strategy":  strategy,
                "direction": "MANUAL",
                "main_leg":  {
                    "strike":      strike,
                    "exp":         expiration,
                    "mid":         entry_price,
                    "option_type": "call"
                },
                "exp": expiration
            },
            "pricing": {
                "entry":     entry_price,
                "stop":      round(entry_price * (1 - STOP_LOSS_PCT), 2),
                "target":    round(entry_price * (1 + PROFIT_TARGET_PCT), 2),
                "contracts": contracts,
            },
            "vol": {},
            "confluence": {}
        }
        return self.open_position(mock_result)

    # ─── Accessors ───────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> Dict[int, Dict]:
        return dict(self._open)

    @property
    def open_count(self) -> int:
        return len(self._open)

    def get_position(self, position_id: int) -> Optional[Dict]:
        return self._open.get(position_id)

    def get_positions_for_ticker(self, ticker: str) -> List[Dict]:
        return [
            p for p in self._open.values()
            if p["ticker"].upper() == ticker.upper()
        ]

    def is_open(self, ticker: str) -> bool:
        return any(
            p["ticker"].upper() == ticker.upper()
            for p in self._open.values()
        )

    # ─── Performance ─────────────────────────────────────────────────────────

    def unrealized_pnl(self, position_id: int, current_price: float) -> float:
        pos = self._open.get(position_id)
        if not pos:
            return 0.0
        contracts  = pos.get("contracts", 1)
        entry_cost = pos.get("entry_cost", 0)
        return current_price * 100 * contracts - entry_cost

    def unrealized_r(self, position_id: int, current_price: float) -> float:
        pos = self._open.get(position_id)
        if not pos:
            return 0.0
        entry_cost = pos.get("entry_cost", 1)
        pnl = self.unrealized_pnl(position_id, current_price)
        return pnl / entry_cost

    def print_summary(self):
        """Print a human-readable summary of open positions to terminal."""
        if not self._open:
            print("  No open positions.")
            return
        print(f"  {'ID':<5} {'Ticker':<8} {'Strategy':<20} {'Entry':>8} "
              f"{'Stop':>8} {'Target':>8} {'DTE':>5} {'Since':<16}")
        print("  " + "─" * 80)
        for pid, pos in self._open.items():
            entry_time = pos.get("entry_time", "")[:16]
            exp = pos.get("expiration", "?")
            dte = "?"
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                dte = str((exp_dt - datetime.now()).days)
            except Exception:
                pass
            print(
                f"  #{pid:<4} {pos['ticker']:<8} "
                f"{(pos.get('strategy') or '?'):<20} "
                f"${pos.get('entry_price', 0):>7.2f} "
                f"${pos.get('stop_price', 0):>7.2f} "
                f"${pos.get('target_price', 0):>7.2f} "
                f"{dte:>5} {entry_time:<16}"
            )
