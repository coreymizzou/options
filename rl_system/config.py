"""
=============================================================================
config.py — Central Configuration
=============================================================================
All tunable parameters live here. Edit this file to adjust behavior
without touching core logic in any other module.
=============================================================================
"""

# ─── Account & Risk ──────────────────────────────────────────────────────────
ACCOUNT_SIZE         = 25_000.0   # Total account size ($)
MAX_RISK_PCT         = 0.02       # Max risk per trade as fraction of account
MAX_RISK_DOLLARS     = ACCOUNT_SIZE * MAX_RISK_PCT   # = $500
MAX_CONCURRENT_POSITIONS = 3      # Hard cap on simultaneous open positions

# ─── Position Management ─────────────────────────────────────────────────────
STOP_LOSS_PCT        = 0.50       # Close position if down this fraction of entry cost
PROFIT_TARGET_PCT    = 1.00       # Take profit at this fraction gain (100% = 2x)
MAX_DTE_AT_ENTRY     = 60         # Never enter with more than this many days to expiry
MIN_DTE_AT_ENTRY     = 21         # Never enter with fewer than this many days to expiry
CLOSE_BEFORE_DTE     = 7          # Force-close any position within this many DTE

# ─── Cooldown / No-Chase Rules ───────────────────────────────────────────────
COOLDOWN_HOURS       = 24         # Hours after close before re-entering same ticker
NO_REENTRY_WHILE_OPEN = True      # Block new entry if position already open on ticker

# ─── 60-Second Loop ──────────────────────────────────────────────────────────
LOOP_INTERVAL_SECONDS = 60        # How often the main loop ticks
SCANNER_RUN_INTERVAL  = 300       # How often to re-run the full options scanner (seconds)
                                  # Scanner is expensive — default every 5 minutes

# ─── Notification / Alerting ─────────────────────────────────────────────────
NOTIFY_CONFIDENCE_THRESHOLD = 0.60   # Min confidence to send user-facing alert
NOTIFY_ON_ACTION_CHANGE     = True   # Alert when recommended action changes
NOTIFY_FORCE_EXIT_ALWAYS    = True   # Always alert on urgent/risk exits
NOTIFY_TERMINAL             = True   # Print alerts to terminal
NOTIFY_WINDOWS_TOAST        = True   # Windows desktop toast notifications
NOTIFY_DISCORD_WEBHOOK_URL  = ""     # Set to Discord webhook URL to enable; leave "" to disable

# ─── Confidence & Decision Thresholds ────────────────────────────────────────
MIN_CONFLUENCE_SCORE_TO_ENTER = 9    # Scanner confluence score floor for entry consideration
ENTER_CONFIDENCE_THRESHOLD   = 0.55  # Agent confidence required to recommend ENTER
EXIT_CONFIDENCE_THRESHOLD    = 0.55  # Agent confidence required to recommend EXIT
HOLD_IS_DEFAULT              = True  # When uncertain, default to HOLD not EXIT

# ─── Drawdown Controls ───────────────────────────────────────────────────────
MAX_DAILY_DRAWDOWN_PCT  = 0.06   # Force no new entries if daily P&L down > 6% of account
MAX_ROLLING_DRAWDOWN_PCT = 0.10  # Penalty trigger for rolling drawdown in reward calc

# ─── Reward Function Weights ─────────────────────────────────────────────────
# reward = base_R - stop_penalty - drawdown_penalty - churn_penalty
REWARD_STOP_PENALTY      = 0.5   # Extra penalty (in R) for hitting stop loss
REWARD_DRAWDOWN_PENALTY  = 0.3   # Penalty for breaching drawdown threshold
REWARD_CHURN_PENALTY     = 0.2   # Penalty per unnecessary trade (overtrading)
REWARD_MIN_HOLD_TICKS    = 5     # Minimum ticks held before exit counts as non-churn

# ─── Learning Layer ──────────────────────────────────────────────────────────
AGENT_LEARNING_RATE      = 0.05   # Online update step size (lower = slower but more stable)
AGENT_EXPLORATION_RATE   = 0.10   # Fraction of ticks using random exploration
AGENT_MIN_SAMPLES_TO_LEARN = 10   # Min closed trades before weights deviate from prior
AGENT_WEIGHT_DECAY       = 0.001  # L2 regularization to prevent overfitting
AGENT_SAVE_INTERVAL_TICKS = 10    # Save weights to DB every N ticks

# ─── Feature Engineering ─────────────────────────────────────────────────────
# These features are computed each tick for open positions and entry candidates
FEATURE_NAMES = [
    "unrealized_r",           # Current unrealized P&L in R-multiples
    "dte_fraction",           # DTE remaining / DTE at entry (0=expiry, 1=just entered)
    "theta_decay_fraction",   # Theta paid so far / total premium at entry
    "iv_rank_normalized",     # IVR / 100
    "spy_trend",              # +1 bullish, 0 neutral, -1 bearish
    "rsi_normalized",         # RSI / 100
    "flow_score_normalized",  # Flow confluence score / 15
    "above_vwap",             # 1 if above VWAP, 0 if below
    "regime_score",           # 1 trending, 0 ranging, -1 risk_off
    "ticks_held_normalized",  # Ticks held / 100 (normalized)
    "spread_vs_target",       # (current_value - entry) / (target - entry)
    "days_since_entry_norm",  # Days since entry / 30
]

# ─── Logging & Debug ─────────────────────────────────────────────────────────
DEBUG_MODE           = False     # Extra verbose output when True
LOG_EVERY_TICK       = True      # Log internal state every tick (to DB, not terminal)
LOG_DIR              = "./logs"  # Directory for log files
DB_PATH              = "./scanner_data.db"  # SQLite database path

# ─── Phase 2 Hooks (not active yet) ──────────────────────────────────────────
# Set to True when broker integration is ready
BROKER_AUTO_EXECUTE  = False     # Never auto-execute in phase 1
BROKER_API_KEY       = ""        # Tradier live account API key (for future use)
BROKER_ACCOUNT_ID    = ""        # Tradier account ID (for future use)
