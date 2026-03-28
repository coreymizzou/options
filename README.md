# Options Scanner — RL Decision Layer

A reinforcement learning decision system that wraps your options scanner with a 60-second evaluation loop, position tracking, hard risk rules, and an online-learning agent that improves over time.

---

## File Structure

```
your_folder/
├── options_scanner.py          ← main scanner (add new tickers here)
│
└── rl_system/
    ├── config.py               ← ALL tunable parameters — start here
    ├── database.py             ← SQLite persistence layer
    ├── notifier.py             ← Terminal + toast + Discord alerts
    ├── position_tracker.py     ← Position lifecycle + hard risk rules
    ├── rl_agent.py             ← Contextual bandit decision agent
    ├── run.py                  ← Main orchestration loop
    └── requirements_rl.txt     ← pip dependencies
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r rl_system/requirements_rl.txt

# 2. Set API keys (Mac/Linux)
export TRADIER_API_KEY="your_key"
export FRED_API_KEY="your_key"

# Windows (Command Prompt)
set TRADIER_API_KEY=your_key
set FRED_API_KEY=your_key

# Windows permanent (recommended)
# Search "Environment Variables" in Windows settings → add under User Variables

# 3. Run from your main folder (where options_scanner.py lives)
cd your_folder
python3 rl_system/run.py --paper
```

---

## CLI Commands

```bash
# Run the loop
python3 rl_system/run.py --paper      # paper trading mode (recommended)
python3 rl_system/run.py              # live mode
python3 rl_system/run.py --debug      # verbose output

# Status and diagnostics
python3 rl_system/run.py --status     # full status: positions, P&L, performance
python3 rl_system/run.py --weights    # agent weight summary

# Position management
python3 rl_system/run.py --close 1    # mark position #1 as manually closed
python3 rl_system/run.py --close-all  # close all open positions
python3 rl_system/run.py --delete 1   # delete position #1 (unfilled order, no record kept)

# Reset
python3 rl_system/run.py --reset      # clear positions/history, keep agent weights
python3 rl_system/run.py --reset-all  # wipe everything including agent weights
```

---

## Typical Daily Workflow

```
1. Start loop at 10am:
   python3 rl_system/run.py --paper

2. Watch terminal for ENTER alerts:
   → Check ThinkorSwim for the recommended spread
   → Place the order
   → Wait for green FILLED confirmation in ThinkorSwim
   → Say y to track (ONLY after confirmed fill — not on working orders)
   → Enter your actual fill price from the Filled Orders tab

3. Monitor with tick status bar (prints every 60 seconds):
   [10:05 ET] tick=12 RISK_OFF VIX=31  |  CRWD#1 +$124 +0.25R  |  MU#2 +$47 +0.09R

4. When EXIT or TARGET_HIT alert fires:
   → Close in ThinkorSwim first
   → Confirm fill price in terminal

5. End of day:
   python3 rl_system/run.py --status    # review performance
   python3 rl_system/run.py --reset     # clear for next day (keeps weights)
```

---

## How It Works

### Every 60 seconds (one tick)

1. **Hard rules evaluated first** — stop loss, profit target, DTE expiry. Always fire, cannot be overridden by the agent.
2. **Open positions scored** — agent outputs HOLD or EXIT with confidence. Alerts only if action changed AND confidence ≥ 0.60.
3. **New candidates scored** — agent outputs ENTER or WAIT for top 5 scanner results. Alerts only on meaningful signals.
4. **Tick status bar printed** — one-line summary of time, regime, VIX, and all open position P&L.

### Every 5 minutes

- Full scanner runs on all watchlist tickers (~2 minutes)
- Results sorted by confluence score
- OI compared to previous scan for flow validation

### When you confirm a fill

- Position recorded with your actual fill price
- 10-minute grace period begins (no exits evaluated during this window)
- Tier 2 checks logged (earnings proximity, sector correlation, OI)

### When a position closes

- Reward computed: `realized_R - stop_penalty - drawdown_penalty - churn_penalty`
- Agent weights updated via online gradient descent
- 24-hour cooldown set on that ticker

---

## Hard Risk Rules

These fire regardless of agent confidence. Cannot be disabled.

| Rule | Default | Config Key |
|------|---------|------------|
| Stop loss | Down 50% from entry | `STOP_LOSS_PCT` |
| Profit target | Up 100% (2x) | `PROFIT_TARGET_PCT` |
| DTE force-close | 7 days to expiry | `CLOSE_BEFORE_DTE` |
| Max concurrent | 3 positions | `MAX_CONCURRENT_POSITIONS` |
| Daily drawdown | Down 6% of account | `MAX_DAILY_DRAWDOWN_PCT` |
| Cooldown | 24 hours post-close | `COOLDOWN_HOURS` |

**Grace period:** Hard rules suppressed for the first 10 minutes after entry to allow price data to stabilize.

**Price sanity check:** If the fetched price is >2.2x or <0.15x of entry price in the first 30 minutes, hard rules skip that tick to protect against bad data. After 30 minutes the window widens to 3.0x.

---

## Tier 1 Features

### Time-of-Day Filter

New ENTER recommendations blocked before 10:00am ET and after 3:30pm ET. Options spreads have wide bid/ask outside this window. Hard exits always fire regardless of time.

```python
MARKET_OPEN_HOUR     = 10
MARKET_CLOSE_HOUR    = 15
MARKET_CLOSE_MINUTE  = 30
ENFORCE_MARKET_HOURS = True   # set False to disable
```

### Action State Expiry

Action states expire after 20 hours. If CRWD was recommended ENTER yesterday but you didn't take it, the state resets overnight so today's fresh signal fires normally.

```python
ACTION_STATE_EXPIRY_HOURS = 20
```

### Live Tick Status Bar

Every tick prints a compact one-liner showing regime, VIX, and all open position P&L:

```
[10:05 ET] tick=12 RISK_OFF VIX=31  |  CRWD#1 +$124 +0.25R  |  MU#2 +$47 +0.09R
```

Suppressed overnight when no positions are open to avoid spam.

---

## Tier 2 Features

### Earnings Calendar

Checks next earnings date via Tradier API (primary) or yfinance (fallback). Cached for 6 hours.

- **Warning zone** (within 5 days): shown in alert — IV crush risk flagged
- **Block zone** (within 2 days): entry hard blocked

```python
EARNINGS_WARN_DAYS    = 5    # must be > BLOCK_DAYS
EARNINGS_BLOCK_DAYS   = 2
EARNINGS_CHECK_ENABLED = True
```

### Sector Correlation Check

Warns when adding a position would concentrate too much in one sector or direction. Warns but does not hard block.

```python
MAX_SAME_SECTOR_POSITIONS    = 2
MAX_SAME_DIRECTION_POSITIONS = 2
SECTOR_CORRELATION_ENABLED   = True
```

Sectors tracked: semiconductors, mega_tech, cybersecurity, cloud_software, media_tech, ev_auto, crypto_tech, financials, energy, commodities, intl_growth, index.

### OI Change Detection

Compares open interest between scanner runs to validate flow direction.

- OI increasing → confirms opening flow → full confidence
- OI flat → may be closing → flagged
- OI decreasing → likely closing flow → confidence reduced 25%

```python
OI_CHANGE_ENABLED    = True
OI_INCREASE_REQUIRED = True
```

---

## The Learning Agent

### Blending schedule

```
0-9 closed trades:    pure rule-based (scanner signals)
10+ closed trades:    rule-based blended with learned model (up to 60% learned)
```

### 12 features per decision

| Feature | What it measures |
|---------|-----------------|
| `unrealized_r` | Current P&L in R-multiples |
| `dte_fraction` | Time remaining / time at entry |
| `theta_decay_fraction` | How much premium theta has eaten |
| `iv_rank_normalized` | IVR / 100 |
| `spy_trend` | +1 bull, 0 neutral, -1 bear |
| `rsi_normalized` | RSI / 100 |
| `flow_score_normalized` | Flow score / 15 |
| `above_vwap` | 1 if price above VWAP |
| `regime_score` | 1 trending, 0 ranging, -1 risk_off |
| `ticks_held_normalized` | Ticks held / 100 |
| `spread_vs_target` | Progress toward profit target |
| `days_since_entry_norm` | Days held / 30 |

### Reward function

```
reward = realized_R
       - 0.5  (if stop loss hit)
       - 0.3  (if rolling drawdown > 10% of account)
       - 0.2  (if held < 5 ticks and lost money)
```

### Exploration

The agent randomly explores 10% of entry decisions early on to gather diverse training data. Rate decays as it learns — halves every 25 updates, floors at 1%. Exploration ticks never fire user-facing alerts.

---

## Confidence Explained

```
conf=0.00 - 0.59  →  below threshold — silent (HOLD / WAIT)
conf=0.60 - 0.79  →  alert fires
conf=0.80 - 0.99  →  high conviction alert
conf=1.00          →  hard rule fired — always alerts
```

**HOLD conf=0.00** means the agent sees zero reason to exit. It is NOT a signal to close. HOLD is the default when exit confidence is low.

---

## Price Fetching for Open Positions

The system fetches the current price of your **exact held contract** (by strike, expiration, and option type) from Tradier every 30 seconds. It does not use the scanner's recommended contract — that would be a different strike.

Fallback chain:
1. Tradier live bid/ask mid
2. yfinance option chain mid
3. Entry price (last resort — P&L shows $0, no exits fire)

---

## Watchlist (19 tickers)

| Ticker | Sector |
|--------|--------|
| NVDA, AMD, MU | Semiconductors |
| AAPL, MSFT, GOOGL, META, AMZN | Mega-cap tech |
| CRWD | Cybersecurity |
| CRM, PLTR | Cloud software |
| NFLX | Media tech |
| TSLA | EV / Auto |
| COIN, MSTR | Crypto-adjacent |
| GS | Financials |
| XOM | Energy |
| GLD | Commodities |
| MELI | International growth |

To add tickers: add to `WATCHLIST` in `options_scanner.py` AND to `SECTOR_MAP` in `config.py`.

---

## Notifications

| Channel | How to enable |
|---------|--------------|
| Terminal | Always on |
| Windows toast | `pip install winotify` |
| Mac notifications | `pip install pyobjus` — or disable: `NOTIFY_WINDOWS_TOAST = False` |
| Discord | Set `NOTIFY_DISCORD_WEBHOOK_URL` in config.py |

---

## Database

All data persists in `scanner_data.db` (SQLite).

| Table | Contents |
|-------|----------|
| `positions` | All open and closed positions with P&L |
| `tick_snapshots` | Every 60-second evaluation per position |
| `recommendations` | Every user-facing alert sent |
| `trade_journal` | Human-readable event log |
| `agent_weights` | Learned model weights |
| `cooldowns` | Active ticker cooldowns |
| `system_state` | Action state and persistent loop state |

```bash
sqlite3 scanner_data.db

# Open positions
SELECT ticker, strategy, entry_price, stop_price, target_price
FROM positions WHERE status='OPEN';

# Closed trade history
SELECT ticker, realized_r, realized_pnl, exit_reason, exit_time
FROM positions WHERE status='CLOSED' ORDER BY exit_time DESC;

# Today's P&L
SELECT SUM(realized_pnl) FROM positions
WHERE status='CLOSED' AND exit_time LIKE '2026-03-28%';

# Recent decisions
SELECT ticker, action, confidence, reason_summary
FROM trade_journal ORDER BY timestamp DESC LIMIT 20;
```

---

## Architecture Notes

- Hard risk rules in `position_tracker.py` cannot be overridden by the agent under any circumstances
- Three-layer false-exit protection: 10-minute grace period + correct contract pricing + price sanity check
- Agent learning is additive — `--reset` clears trade history but keeps weights
- All state (positions, weights, action states, cooldowns) survives a restart
- Weights saved every 10 ticks and on every position close
- No auto-execution — system recommends, you execute in ThinkorSwim, then confirm fill

---

## Troubleshooting

**"OUTSIDE market hours" at startup** — correct if before 10am or after 3:30pm ET. No entries until market hours.

**"Windows toast failed: No usable implementation"** — install `winotify` (Windows) or `pyobjus` (Mac), or set `NOTIFY_WINDOWS_TOAST = False` in config.py. Terminal alerts still work.

**System suggests closing right after opening** — should not happen due to 10-minute grace period. If it does, check that the Tradier API key is active and returning data.

**Loop recommends same trade repeatedly** — action states expire after 20 hours. If repeating within one session, check that action state is updating correctly (look for "action state set to HOLD" in logs after tracking).

**Scanner not importing** — run from the folder containing `options_scanner.py`, not from inside `rl_system/`. The path must be: `python3 rl_system/run.py` not `python3 run.py`.

**Tradier price fetch failing** — verify key is set: `echo $TRADIER_API_KEY` (Mac) or `echo %TRADIER_API_KEY%` (Windows). Paper trading key is separate from live trading key.
