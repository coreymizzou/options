"""
=============================================================================
database.py — SQLite Persistence Layer
=============================================================================
Handles all reads and writes to the local SQLite database.
Schema covers: positions, recommendations, market snapshots,
trade journal, decision history, and agent weights.
=============================================================================
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import DB_PATH, LOG_DIR

# ─── Numpy-safe JSON encoder ─────────────────────────────────────────────────
# numpy float32/int64 values are not natively JSON serializable.
# This encoder converts them to standard Python types transparently.
class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        # Fallback: convert anything else to string
        # handles datetime, Path, and other non-serializable types
        try:
            return str(obj)
        except Exception:
            return super().default(obj)

def _dumps(obj) -> str:
    """json.dumps with numpy type support."""
    return json.dumps(obj, cls=_SafeEncoder)

# ─── Logging setup ───────────────────────────────────────────────────────────
Path(LOG_DIR).mkdir(exist_ok=True)
logger = logging.getLogger(__name__)


# =============================================================================
#  SCHEMA
# =============================================================================

SCHEMA_SQL = """
-- ── Positions ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    strategy            TEXT,
    direction           TEXT,
    option_type         TEXT,
    strike              REAL,
    expiration          TEXT,
    entry_price         REAL,       -- per share (mid at entry)
    entry_cost          REAL,       -- total dollars risked (entry_price * 100 * contracts)
    contracts           INTEGER,
    stop_price          REAL,
    target_price        REAL,
    entry_time          TEXT,
    exit_time           TEXT,
    exit_price          REAL,
    exit_reason         TEXT,       -- STOP_HIT / TARGET_HIT / MANUAL / DTE_EXPIRY / AGENT_EXIT
    realized_pnl        REAL,
    realized_r          REAL,       -- P&L in R-multiples (pnl / initial_risk)
    status              TEXT DEFAULT 'OPEN',  -- OPEN / CLOSED
    confluence_score    INTEGER,
    entry_ivr           REAL,
    entry_dte           INTEGER,
    notes               TEXT,
    raw_scanner_data    TEXT        -- JSON blob of full scanner result at entry
);

-- ── Tick Snapshots ────────────────────────────────────────────────────────────
-- One row per position per 60-second tick while open
CREATE TABLE IF NOT EXISTS tick_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id         INTEGER REFERENCES positions(id),
    ticker              TEXT,
    timestamp           TEXT,
    current_price       REAL,       -- underlying spot
    option_mid          REAL,       -- current option/spread mid price
    unrealized_pnl      REAL,
    unrealized_r        REAL,
    dte_remaining       INTEGER,
    theta_today         REAL,
    iv_current          REAL,
    ivr_current         REAL,
    spy_change_pct      REAL,
    rsi                 REAL,
    above_vwap          INTEGER,    -- 0/1
    regime              TEXT,
    flow_score          INTEGER,
    feature_vector      TEXT,       -- JSON array of normalized features
    agent_action        TEXT,       -- ENTER / HOLD / EXIT / WAIT
    agent_confidence    REAL,
    agent_reasons       TEXT        -- JSON array of top reason strings
);

-- ── Recommendations ───────────────────────────────────────────────────────────
-- Every recommendation surfaced to the user
CREATE TABLE IF NOT EXISTS recommendations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT,
    ticker              TEXT,
    strategy            TEXT,
    direction           TEXT,
    strike              REAL,
    expiration          TEXT,
    action              TEXT,       -- ENTER / HOLD / EXIT / WAIT
    confidence          REAL,
    reasons             TEXT,       -- JSON array
    market_snapshot     TEXT,       -- JSON blob
    notified_user       INTEGER DEFAULT 0,  -- 0/1
    acted_upon          INTEGER DEFAULT 0,  -- set to 1 when position opened/closed
    position_id         INTEGER REFERENCES positions(id)
);

-- ── Broker Orders ───────────────────────────────────────────────────────────
-- Durable order lifecycle state. The runtime still keeps a small pending queue
-- for fast reconciliation, but this table is the auditable source of broker
-- order status across restarts.
CREATE TABLE IF NOT EXISTS orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id     TEXT UNIQUE,
    client_order_id     TEXT,
    order_kind          TEXT,       -- ENTRY / EXIT
    status              TEXT,       -- SUBMITTED / FILLED / CANCELLED / REJECTED / EXPIRED
    ticker              TEXT,
    position_id         INTEGER REFERENCES positions(id),
    side                TEXT,
    strategy            TEXT,
    option_type         TEXT,
    strike              REAL,
    expiration          TEXT,
    quantity            INTEGER,
    limit_price         REAL,
    filled_price        REAL,
    submitted_at        TEXT,
    updated_at          TEXT,
    raw_payload         TEXT,
    notes               TEXT
);

-- ── Candidate Evaluations ───────────────────────────────────────────────────
-- Every scanner candidate that survives hard pre-filters, including WAITs.
-- Future outcome fields are intentionally nullable so a later job can label
-- skipped trades with 1h/EOD/1d/MFE/MAE proxy outcomes.
CREATE TABLE IF NOT EXISTS candidate_evaluations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT,
    ticker              TEXT,
    strategy            TEXT,
    direction           TEXT,
    strike              REAL,
    expiration          TEXT,
    action              TEXT,
    confidence          REAL,
    confluence_score    INTEGER,
    flow_confidence     TEXT,
    ivr                 REAL,
    regime              TEXT,
    above_vwap          INTEGER,
    entry_price         REAL,
    reasons             TEXT,
    market_snapshot     TEXT,
    raw_scanner_data    TEXT,
    recommendation_id   INTEGER REFERENCES recommendations(id),
    position_id         INTEGER REFERENCES positions(id),
    outcome_1h_r        REAL,
    outcome_eod_r       REAL,
    outcome_1d_r        REAL,
    mfe_r               REAL,
    mae_r               REAL,
    outcome_labeled_at  TEXT
);

-- ── Candidate Price Snapshots ───────────────────────────────────────────────
-- Timestamped forward prices for candidate_evaluations. Outcome labels must be
-- derived from these snapshots, not from "current quote when labeler ran".
CREATE TABLE IF NOT EXISTS candidate_price_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id         INTEGER REFERENCES candidate_evaluations(id),
    timestamp            TEXT,
    option_price         REAL,
    underlying_price     REAL,
    source               TEXT,
    raw_payload          TEXT
);

CREATE INDEX IF NOT EXISTS idx_candidate_price_snapshots_candidate_time
    ON candidate_price_snapshots(candidate_id, timestamp);

-- ── Trade Journal ─────────────────────────────────────────────────────────────
-- Human-readable decision log — one entry per notable event
CREATE TABLE IF NOT EXISTS trade_journal (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT,
    event_type          TEXT,       -- ENTRY_REC / EXIT_REC / POSITION_OPENED /
                                    -- POSITION_CLOSED / ALERT_SENT / COOLDOWN /
                                    -- RISK_BLOCK / AGENT_UPDATE / SYSTEM_START /
                                    -- SYSTEM_STOP
    ticker              TEXT,
    position_id         INTEGER,
    action              TEXT,
    confidence          REAL,
    reason_summary      TEXT,       -- plain-English summary
    details             TEXT        -- JSON blob with full context
);

-- ── Agent Weights ─────────────────────────────────────────────────────────────
-- Persisted learning layer parameters — one row per named weight set
CREATE TABLE IF NOT EXISTS agent_weights (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT,
    label               TEXT,       -- e.g. 'enter_weights', 'exit_weights'
    weights             TEXT,       -- JSON array of floats
    bias                REAL,
    n_updates           INTEGER,    -- how many online updates applied
    mean_reward         REAL,       -- running mean reward seen so far
    notes               TEXT
);

-- ── Cooldown State ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cooldowns (
    ticker              TEXT PRIMARY KEY,
    last_close_time     TEXT,
    cooldown_until      TEXT,
    reason              TEXT
);

-- ── System State ─────────────────────────────────────────────────────────────
-- Key-value store for misc persistent state
CREATE TABLE IF NOT EXISTS system_state (
    key                 TEXT PRIMARY KEY,
    value               TEXT,
    updated_at          TEXT
);
"""


# =============================================================================
#  CONNECTION MANAGER
# =============================================================================

def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent write performance
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database():
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
    logger.info(f"Database initialized at {DB_PATH}")


# =============================================================================
#  POSITIONS
# =============================================================================

def insert_position(data: Dict[str, Any]) -> int:
    """Insert a new open position. Returns the new position ID."""
    sql = """
        INSERT INTO positions (
            ticker, strategy, direction, option_type, strike, expiration,
            entry_price, entry_cost, contracts, stop_price, target_price,
            entry_time, status, confluence_score, entry_ivr, entry_dte,
            notes, raw_scanner_data
        ) VALUES (
            :ticker, :strategy, :direction, :option_type, :strike, :expiration,
            :entry_price, :entry_cost, :contracts, :stop_price, :target_price,
            :entry_time, 'OPEN', :confluence_score, :entry_ivr, :entry_dte,
            :notes, :raw_scanner_data
        )
    """
    with get_connection() as conn:
        cur = conn.execute(sql, data)
        return cur.lastrowid


def close_position(position_id: int, exit_price: float, exit_reason: str,
                   realized_pnl: float, realized_r: float):
    """Mark a position as closed."""
    sql = """
        UPDATE positions SET
            exit_time    = :exit_time,
            exit_price   = :exit_price,
            exit_reason  = :exit_reason,
            realized_pnl = :realized_pnl,
            realized_r   = :realized_r,
            status       = 'CLOSED'
        WHERE id = :id
    """
    with get_connection() as conn:
        conn.execute(sql, {
            "id": position_id,
            "exit_time": datetime.now().isoformat(),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "realized_pnl": realized_pnl,
            "realized_r": realized_r
        })


def get_open_positions() -> List[Dict]:
    """Return all currently open positions as list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_time"
        ).fetchall()
    return [dict(r) for r in rows]


def get_position_by_id(position_id: int) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
    return dict(row) if row else None


def get_closed_positions(limit: int = 100) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY exit_time DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_position_notes(position_id: int, notes: str):
    with get_connection() as conn:
        conn.execute(
            "UPDATE positions SET notes = ? WHERE id = ?", (notes, position_id)
        )


# =============================================================================
#  TICK SNAPSHOTS
# =============================================================================

def insert_tick_snapshot(data: Dict[str, Any]) -> int:
    sql = """
        INSERT INTO tick_snapshots (
            position_id, ticker, timestamp, current_price, option_mid,
            unrealized_pnl, unrealized_r, dte_remaining, theta_today,
            iv_current, ivr_current, spy_change_pct, rsi, above_vwap,
            regime, flow_score, feature_vector, agent_action,
            agent_confidence, agent_reasons
        ) VALUES (
            :position_id, :ticker, :timestamp, :current_price, :option_mid,
            :unrealized_pnl, :unrealized_r, :dte_remaining, :theta_today,
            :iv_current, :ivr_current, :spy_change_pct, :rsi, :above_vwap,
            :regime, :flow_score, :feature_vector, :agent_action,
            :agent_confidence, :agent_reasons
        )
    """
    with get_connection() as conn:
        cur = conn.execute(sql, data)
        return cur.lastrowid


def get_recent_snapshots(position_id: int, limit: int = 20) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM tick_snapshots WHERE position_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (position_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
#  RECOMMENDATIONS
# =============================================================================

def insert_recommendation(data: Dict[str, Any]) -> int:
    sql = """
        INSERT INTO recommendations (
            timestamp, ticker, strategy, direction, strike, expiration,
            action, confidence, reasons, market_snapshot, notified_user,
            position_id
        ) VALUES (
            :timestamp, :ticker, :strategy, :direction, :strike, :expiration,
            :action, :confidence, :reasons, :market_snapshot, :notified_user,
            :position_id
        )
    """
    with get_connection() as conn:
        cur = conn.execute(sql, data)
        return cur.lastrowid


def mark_recommendation_notified(rec_id: int):
    with get_connection() as conn:
        conn.execute(
            "UPDATE recommendations SET notified_user = 1 WHERE id = ?", (rec_id,)
        )


def mark_recommendation_acted(rec_id: int):
    with get_connection() as conn:
        conn.execute(
            "UPDATE recommendations SET acted_upon = 1 WHERE id = ?", (rec_id,)
        )


def get_recent_recommendations(limit: int = 20) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
#  ORDERS
# =============================================================================

def upsert_order(data: Dict[str, Any]) -> int:
    """Insert or update a broker order lifecycle row."""
    now = datetime.now().isoformat()
    payload = {
        "broker_order_id": data.get("broker_order_id"),
        "client_order_id": data.get("client_order_id"),
        "order_kind": data.get("order_kind"),
        "status": data.get("status", "SUBMITTED"),
        "ticker": data.get("ticker"),
        "position_id": data.get("position_id"),
        "side": data.get("side"),
        "strategy": data.get("strategy"),
        "option_type": data.get("option_type"),
        "strike": data.get("strike"),
        "expiration": data.get("expiration"),
        "quantity": data.get("quantity"),
        "limit_price": data.get("limit_price"),
        "filled_price": data.get("filled_price"),
        "submitted_at": data.get("submitted_at") or now,
        "updated_at": now,
        "raw_payload": _dumps(data.get("raw_payload") or {}),
        "notes": data.get("notes", ""),
    }
    sql = """
        INSERT INTO orders (
            broker_order_id, client_order_id, order_kind, status, ticker,
            position_id, side, strategy, option_type, strike, expiration,
            quantity, limit_price, filled_price, submitted_at, updated_at,
            raw_payload, notes
        ) VALUES (
            :broker_order_id, :client_order_id, :order_kind, :status, :ticker,
            :position_id, :side, :strategy, :option_type, :strike, :expiration,
            :quantity, :limit_price, :filled_price, :submitted_at, :updated_at,
            :raw_payload, :notes
        )
        ON CONFLICT(broker_order_id) DO UPDATE SET
            client_order_id = excluded.client_order_id,
            order_kind      = excluded.order_kind,
            status          = excluded.status,
            ticker          = excluded.ticker,
            position_id     = COALESCE(excluded.position_id, orders.position_id),
            side            = excluded.side,
            strategy        = excluded.strategy,
            option_type     = excluded.option_type,
            strike          = excluded.strike,
            expiration      = excluded.expiration,
            quantity        = excluded.quantity,
            limit_price     = excluded.limit_price,
            filled_price    = COALESCE(excluded.filled_price, orders.filled_price),
            updated_at      = excluded.updated_at,
            raw_payload     = excluded.raw_payload,
            notes           = excluded.notes
    """
    with get_connection() as conn:
        cur = conn.execute(sql, payload)
        row = conn.execute(
            "SELECT id FROM orders WHERE broker_order_id = ?",
            (payload["broker_order_id"],)
        ).fetchone()
    return int(row["id"]) if row else int(cur.lastrowid)


def update_order_status(
    broker_order_id: str,
    status: str,
    filled_price: float = None,
    position_id: int = None,
    raw_payload: dict = None,
    notes: str = None,
):
    """Update durable broker order state."""
    sets = ["status = ?", "updated_at = ?"]
    vals: List[Any] = [status, datetime.now().isoformat()]
    if filled_price is not None:
        sets.append("filled_price = ?")
        vals.append(filled_price)
    if position_id is not None:
        sets.append("position_id = ?")
        vals.append(position_id)
    if raw_payload is not None:
        sets.append("raw_payload = ?")
        vals.append(_dumps(raw_payload))
    if notes is not None:
        sets.append("notes = ?")
        vals.append(notes)
    vals.append(broker_order_id)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE orders SET {', '.join(sets)} WHERE broker_order_id = ?",
            vals
        )


def get_order_by_broker_id(broker_order_id: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE broker_order_id = ?",
            (broker_order_id,)
        ).fetchone()
    return dict(row) if row else None


# =============================================================================
#  CANDIDATE EVALUATIONS
# =============================================================================

def insert_candidate_evaluation(data: Dict[str, Any]) -> int:
    """Log an evaluated scanner candidate, including WAIT/skipped decisions."""
    sql = """
        INSERT INTO candidate_evaluations (
            timestamp, ticker, strategy, direction, strike, expiration,
            action, confidence, confluence_score, flow_confidence, ivr,
            regime, above_vwap, entry_price, reasons, market_snapshot,
            raw_scanner_data, recommendation_id, position_id
        ) VALUES (
            :timestamp, :ticker, :strategy, :direction, :strike, :expiration,
            :action, :confidence, :confluence_score, :flow_confidence, :ivr,
            :regime, :above_vwap, :entry_price, :reasons, :market_snapshot,
            :raw_scanner_data, :recommendation_id, :position_id
        )
    """
    with get_connection() as conn:
        cur = conn.execute(sql, {
            "timestamp": data.get("timestamp") or datetime.now().isoformat(),
            "ticker": data.get("ticker"),
            "strategy": data.get("strategy"),
            "direction": data.get("direction"),
            "strike": data.get("strike"),
            "expiration": data.get("expiration"),
            "action": data.get("action"),
            "confidence": data.get("confidence"),
            "confluence_score": data.get("confluence_score"),
            "flow_confidence": data.get("flow_confidence"),
            "ivr": data.get("ivr"),
            "regime": data.get("regime"),
            "above_vwap": data.get("above_vwap"),
            "entry_price": data.get("entry_price"),
            "reasons": _dumps(data.get("reasons") or []),
            "market_snapshot": _dumps(data.get("market_snapshot") or {}),
            "raw_scanner_data": _dumps(data.get("raw_scanner_data") or {}),
            "recommendation_id": data.get("recommendation_id"),
            "position_id": data.get("position_id"),
        })
        return cur.lastrowid


def insert_candidate_price_snapshot(data: Dict[str, Any]) -> Optional[int]:
    """Store a timestamped price observation for a candidate evaluation."""
    if not data.get("candidate_id") or data.get("option_price") is None:
        return None
    sql = """
        INSERT INTO candidate_price_snapshots (
            candidate_id, timestamp, option_price, underlying_price, source, raw_payload
        ) VALUES (
            :candidate_id, :timestamp, :option_price, :underlying_price, :source, :raw_payload
        )
    """
    with get_connection() as conn:
        cur = conn.execute(sql, {
            "candidate_id": data.get("candidate_id"),
            "timestamp": data.get("timestamp") or datetime.now().isoformat(),
            "option_price": data.get("option_price"),
            "underlying_price": data.get("underlying_price"),
            "source": data.get("source"),
            "raw_payload": _dumps(data.get("raw_payload") or {}),
        })
        return cur.lastrowid


def get_candidate_snapshot_targets(limit: int = 1000, max_age_hours: float = 30.0) -> List[Dict]:
    """
    Return recent candidates that still need forward price snapshots for labeling.
    """
    sql = """
        SELECT *
        FROM candidate_evaluations
        WHERE raw_scanner_data IS NOT NULL
          AND entry_price IS NOT NULL
          AND expiration IS NOT NULL
          AND (
            outcome_1h_r IS NULL OR outcome_eod_r IS NULL OR outcome_1d_r IS NULL
            OR mfe_r IS NULL OR mae_r IS NULL
          )
        ORDER BY timestamp DESC
        LIMIT ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    targets = []
    for row in rows:
        item = dict(row)
        try:
            if datetime.fromisoformat(item.get("timestamp", "")) < cutoff:
                continue
        except Exception:
            continue
        targets.append(item)
    return targets


def get_candidate_price_snapshots(candidate_id: int) -> List[Dict]:
    """Return all price snapshots for one candidate ordered by timestamp."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM candidate_price_snapshots
            WHERE candidate_id = ?
            ORDER BY timestamp
            """,
            (candidate_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
#  TRADE JOURNAL
# =============================================================================

def log_journal_event(event_type: str, ticker: str = None, position_id: int = None,
                      action: str = None, confidence: float = None,
                      reason_summary: str = None, details: dict = None):
    """Write a human-readable event to the trade journal."""
    sql = """
        INSERT INTO trade_journal (
            timestamp, event_type, ticker, position_id, action,
            confidence, reason_summary, details
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            datetime.now().isoformat(),
            event_type,
            ticker,
            position_id,
            action,
            confidence,
            reason_summary,
            _dumps(details or {})
        ))


def get_journal(limit: int = 50, ticker: str = None) -> List[Dict]:
    with get_connection() as conn:
        if ticker:
            rows = conn.execute(
                """SELECT * FROM trade_journal WHERE ticker = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (ticker, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trade_journal ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
#  AGENT WEIGHTS
# =============================================================================

def save_agent_weights(label: str, weights: list, bias: float,
                       n_updates: int, mean_reward: float, notes: str = ""):
    """Upsert agent weights for a given label."""
    # Delete old entry for this label, insert fresh
    sql_del = "DELETE FROM agent_weights WHERE label = ?"
    sql_ins = """
        INSERT INTO agent_weights (timestamp, label, weights, bias,
                                   n_updates, mean_reward, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql_del, (label,))
        conn.execute(sql_ins, (
            datetime.now().isoformat(),
            label,
            _dumps(weights),
            bias,
            n_updates,
            mean_reward,
            notes
        ))


def load_agent_weights(label: str) -> Optional[Dict]:
    """Load the most recent weights for a given label."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM agent_weights WHERE label = ?", (label,)
        ).fetchone()
    if row:
        d = dict(row)
        d["weights"] = json.loads(d["weights"])
        return d
    return None


# =============================================================================
#  COOLDOWN STATE
# =============================================================================

def set_cooldown(ticker: str, cooldown_until: str, reason: str = ""):
    sql = """
        INSERT INTO cooldowns (ticker, last_close_time, cooldown_until, reason)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            last_close_time = excluded.last_close_time,
            cooldown_until  = excluded.cooldown_until,
            reason          = excluded.reason
    """
    with get_connection() as conn:
        conn.execute(sql, (ticker, datetime.now().isoformat(), cooldown_until, reason))


def get_cooldown(ticker: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM cooldowns WHERE ticker = ?", (ticker,)
        ).fetchone()
    return dict(row) if row else None


def clear_cooldown(ticker: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM cooldowns WHERE ticker = ?", (ticker,))


def is_on_cooldown(ticker: str) -> bool:
    """Returns True if ticker is currently in cooldown period."""
    cd = get_cooldown(ticker)
    if not cd:
        return False
    cooldown_until = datetime.fromisoformat(cd["cooldown_until"])
    return datetime.now() < cooldown_until


# =============================================================================
#  SYSTEM STATE
# =============================================================================

def set_state(key: str, value: Any):
    """Persist a key-value pair to system state."""
    sql = """
        INSERT INTO system_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value      = excluded.value,
            updated_at = excluded.updated_at
    """
    with get_connection() as conn:
        conn.execute(sql, (key, _dumps(value), datetime.now().isoformat()))


def get_state(key: str, default=None) -> Any:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?", (key,)
        ).fetchone()
    if row:
        return json.loads(row["value"])
    return default


# =============================================================================
#  ANALYTICS HELPERS
# =============================================================================

def get_performance_summary() -> Dict:
    """Return high-level P&L stats from closed positions."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                            AS total_trades,
                SUM(CASE WHEN realized_r > 0 THEN 1 ELSE 0 END) AS winners,
                SUM(CASE WHEN realized_r <= 0 THEN 1 ELSE 0 END) AS losers,
                AVG(realized_r)                     AS avg_r,
                SUM(realized_pnl)                   AS total_pnl,
                MAX(realized_r)                     AS best_r,
                MIN(realized_r)                     AS worst_r
            FROM positions WHERE status = 'CLOSED'
        """).fetchone()
    return dict(row) if row else {}


def get_daily_pnl() -> float:
    """Return today's realized P&L from closed positions."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM(realized_pnl), 0) AS pnl
               FROM positions
               WHERE status = 'CLOSED' AND exit_time LIKE ?""",
            (f"{today}%",)
        ).fetchone()
    return float(row["pnl"]) if row else 0.0

def get_open_unrealized_pnl() -> float:
    """Return latest recorded unrealized P&L across open positions."""
    sql = """
        SELECT COALESCE(SUM(ts.unrealized_pnl), 0) AS pnl
        FROM positions p
        LEFT JOIN tick_snapshots ts
          ON ts.id = (
              SELECT id FROM tick_snapshots
              WHERE position_id = p.id
              ORDER BY timestamp DESC
              LIMIT 1
          )
        WHERE p.status = 'OPEN'
    """
    with get_connection() as conn:
        row = conn.execute(sql).fetchone()
    return float(row["pnl"]) if row else 0.0

# =============================================================================
#  PENDING ORDERS PERSISTENCE
# =============================================================================

PENDING_ORDERS_KEY = "pending_orders"
PENDING_EXIT_ORDERS_KEY = "pending_exit_orders"

def save_pending_orders(pending: dict):
    """Persist pending order dict to system_state so restarts don't lose them."""
    set_state(PENDING_ORDERS_KEY, pending)

def load_pending_orders() -> dict:
    """Load pending orders from system_state on startup."""
    return get_state(PENDING_ORDERS_KEY, default={})

def clear_pending_orders():
    """Wipe pending orders from system_state."""
    set_state(PENDING_ORDERS_KEY, {})

def save_pending_exit_orders(pending: dict):
    """Persist pending exit order dict so restarts don't lose close orders."""
    set_state(PENDING_EXIT_ORDERS_KEY, pending)

def load_pending_exit_orders() -> dict:
    """Load pending exit orders from system_state on startup."""
    return get_state(PENDING_EXIT_ORDERS_KEY, default={})

def clear_pending_exit_orders():
    """Wipe pending exit orders from system_state."""
    set_state(PENDING_EXIT_ORDERS_KEY, {})
