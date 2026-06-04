"""
=============================================================================
config.py — Central Configuration
=============================================================================
All tunable parameters live here. Edit this file to adjust behavior
without touching core logic in any other module.
=============================================================================
"""

# ─── Account & Risk ──────────────────────────────────────────────────────────
ACCOUNT_SIZE         = 100_000.0   # Total account size ($)
MAX_RISK_PCT         = 0.02       # Max risk per trade as fraction of account
MAX_RISK_DOLLARS     = ACCOUNT_SIZE * MAX_RISK_PCT   # = $500
MAX_CONCURRENT_POSITIONS = 4      # Hard cap on simultaneous open positions

# ─── Position Management ─────────────────────────────────────────────────────
STOP_LOSS_PCT        = 0.33       # Close position if down this fraction of entry cost
PROFIT_TARGET_PCT    = 0.66       # Take profit at this fraction gain (100% = 2x)
TRAILING_PROFIT_ENABLED = True    # Protect meaningful winners before they round-trip
TRAILING_PROFIT_ACTIVATE_R = 0.35 # Start trailing after position reaches this R
TRAILING_PROFIT_GIVEBACK_R = 0.25 # Exit if it gives back this much R from peak
TRAILING_PROFIT_MIN_LOCK_R = 0.05 # Do not trail-exit below this remaining profit
STALLED_EXIT_ENABLED = True       # Exit positions that fail to make progress
STALLED_EXIT_MIN_DAYS = 5         # Minimum hold before stalled exit can fire
STALLED_EXIT_MAX_ABS_R = 0.10     # Near-flat band after several days
STALLED_EXIT_MAX_MFE_R = 0.20     # Never made a useful move in our favor
MAX_DTE_AT_ENTRY     = 60         # Never enter with more than this many days to expiry
MIN_DTE_AT_ENTRY     = 21         # Never enter with fewer than this many days to expiry
CLOSE_BEFORE_DTE     = 7          # Force-close any position within this many DTE
EOD_CLOSE_CALLS      = True       # EOD close guard for short-dated long premium only
EOD_CLOSE_MAX_DTE    = 1          # Do not EOD-close normal swing trades above this DTE
EOD_CLOSE_HOUR       = 15         # ET hour for short-dated EOD close
EOD_CLOSE_MINUTE     = 20         # ET minute for short-dated EOD close

# ─── Cooldown / No-Chase Rules ───────────────────────────────────────────────
COOLDOWN_HOURS       = 24         # Hours after close before re-entering same ticker
NO_REENTRY_WHILE_OPEN = True      # Block new entry if position already open on ticker

# ─── Time-of-Day Filter ──────────────────────────────────────────────────────
# Suppress new ENTER recommendations outside these ET hours
# Options spreads have wide bid/ask in first 30min and last 15min of session
# Hard exits (stop/target/DTE) always fire regardless of time
MARKET_OPEN_HOUR     = 10         # No new entries before 10:00am ET
MARKET_OPEN_MINUTE   = 0
MARKET_CLOSE_HOUR    = 15         # No new entries after 3:30pm ET
MARKET_CLOSE_MINUTE  = 30
ENFORCE_MARKET_HOURS = True       # Set False to disable filter (e.g. for testing)

# ─── Action State Expiry ─────────────────────────────────────────────────────
# How long before a previously seen action state is considered stale
# Prevents yesterday's ENTER state from suppressing today's fresh signal
ACTION_STATE_EXPIRY_HOURS = 20    # States older than this are ignored (reset to None)

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
# Evaluate MODERATE+ scanner setups, but require stronger agent confidence for
# lower-confluence candidates. This avoids starving the agent while still
# keeping weak scanner output away from auto-execution.
MIN_CONFLUENCE_SCORE_TO_EVALUATE = 9 # Candidate must be very strong to reach agent
MIN_CONFLUENCE_SCORE_TO_ENTER    = 9 # Backward-compatible alias for older code paths
MODERATE_CONFLUENCE_MAX_SCORE    = 6 # Scores <= this need extra confidence
MODERATE_CONFLUENCE_CONFIDENCE_BONUS = 0.07
ENTER_CONFIDENCE_THRESHOLD   = 0.55  # Agent confidence required to recommend ENTER
EXIT_CONFIDENCE_THRESHOLD    = 0.55  # Agent confidence required to recommend EXIT
HOLD_IS_DEFAULT              = True  # When uncertain, default to HOLD not EXIT
LOG_SCANNER_SCORECARD       = True  # Log per-ticker scanner scores each refresh

# ─── Supervised Entry Model ──────────────────────────────────────────────────
ML_ENTRY_MODEL_ENABLED = True
ML_ENTRY_MODEL_PATH = "rl_system/models/entry_model.pkl"
ML_MIN_TRAINING_ROWS = 100
ML_ENTRY_MODEL_OBSERVE_ONLY = True # Log ML predictions, but do not alter live entries yet
ML_MIN_EXPECTED_R = 0.03          # Require positive expected edge when model is active
ML_CONFIDENCE_BLEND = 0.35        # Blend model confidence into agent confidence
CANDIDATE_SNAPSHOT_LOOKBACK_HOURS = 30
MAX_CANDIDATE_SNAPSHOT_TARGETS = 1000
ML_AUTO_MAINTENANCE_ENABLED = True
ML_AUTO_MAINTENANCE_HOUR_ET = 16   # Run once daily after the market closes
ML_AUTO_TARGET_HORIZON = "eod"
ML_AUTO_ALLOW_TARGET_FALLBACKS = True  # Keep training alive until strict EOD labels are plentiful

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
AGENT_EXPLORATION_RATE   = 0.05   # Fraction of early ticks using random exploration
AGENT_MIN_SAMPLES_TO_LEARN = 25   # Min closed trades before weights deviate from prior
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

# ─── Earnings Calendar ───────────────────────────────────────────────────────
EARNINGS_WARN_DAYS   = 5         # Warn if earnings within this many days
EARNINGS_BLOCK_DAYS  = 2         # Block new entries if earnings within this many days
                                  # NOTE: WARN_DAYS must be > BLOCK_DAYS or warn never fires
EARNINGS_CHECK_ENABLED = True    # Set False to disable earnings check

# ─── Sector Correlation ───────────────────────────────────────────────────────
MAX_SAME_SECTOR_POSITIONS = 2    # Max open positions in the same sector
MAX_SAME_DIRECTION_POSITIONS = 3 # Max open positions in the same direction (BULLISH/BEARISH)
SECTOR_CORRELATION_ENABLED = True

# Sector mapping for watchlist tickers
# Used to detect when multiple open positions are correlated
SECTOR_MAP = {
    "NVDA":  "semiconductors",
    "AMD":   "semiconductors",
    "MU":    "semiconductors",
    "AVGO":  "semiconductors",
    "LRCX":  "semiconductors",
    "QCOM":  "semiconductors",
    "AAPL":  "mega_tech",
    "MSFT":  "mega_tech",
    "GOOGL": "mega_tech",
    "META":  "mega_tech",
    "AMZN":  "mega_tech",
    "CRWD":  "cybersecurity",
    "CRM":   "cloud_software",
    "PLTR":  "cloud_software",
    "NFLX":  "media_tech",
    "TSLA":  "ev_auto",
    "COIN":  "crypto_tech",
    "MSTR":  "crypto_tech",
    "SPY":   "index",
    "QQQ":   "index",
    # ── New additions ─────────────────────────────────
    "GS":    "financials",    # Goldman — Fed/rate sensitive, high IV
    "XOM":   "energy",        # ExxonMobil — oil/geo driven, uncorrelated to tech
    "GLD":   "commodities",   # Gold ETF — RISK_OFF hedge, rallies when equities sell
    "MELI":  "intl_growth",   # MercadoLibre — Latin American e-commerce, different catalyst set
}

# ─── OI Change Detection ─────────────────────────────────────────────────────
OI_CHANGE_ENABLED    = True      # Compare OI to previous scan to validate flow
OI_INCREASE_REQUIRED = True      # Require OI to increase to confirm opening flow
                                  # Set False to treat all flow as valid

# ─── Auto Mode ───────────────────────────────────────────────────────────────
AUTO_MAX_POSITIONS   = 4          # Max concurrent positions in --auto mode
AUTO_MAX_CAPITAL_PCT = 0.90       # Max % of account to deploy at once in --auto mode
                                  # e.g. 0.40 = never risk more than $10,000 simultaneously
                                  # Each position still respects MAX_RISK_DOLLARS
AUTO_MAX_CANDIDATES_TO_EVALUATE = 10 # Look past blocked/top-heavy candidates before giving up
MAX_EXECUTABLE_PRICE_DIVERGENCE = 0.25  # Skip entries if marketable limit is >25% from scanner price
SPREAD_ORDER_LIMIT_MARKUP = 0.12  # Cap spread buy limits to net_mid * (1 + markup)
AUTO_MAX_SINGLE_LEG_PREMIUM = 20.00 # Hard live cap per-share; $20 = $2,000 per contract
AUTO_MAX_SINGLE_LEG_COST = 2_000.0  # Hard live cap per single-leg position
AUTO_ALLOWED_TICKERS = {
    "NVDA", "TSLA", "AAPL", "MSFT", "META",
    "AMZN", "GOOGL", "AMD", "MU", "MSTR",
    "COIN", "PLTR", "NFLX", "CRWD", "CRM",
    "XOM", "GLD", "AVGO", "LRCX", "QCOM",
}

# ─── Phase 2 Hooks (not active yet) ──────────────────────────────────────────
# Set to True when broker integration is ready
BROKER_AUTO_EXECUTE  = False     # Never auto-execute in phase 1
BROKER_API_KEY       = ""        # Tradier live account API key (for future use)
BROKER_ACCOUNT_ID    = ""        # Tradier account ID (for future use)
