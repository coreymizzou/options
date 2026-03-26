# Options Scanner — RL Decision Layer

## File Structure

```
your_folder/
├── options_scanner.py          ← existing scanner (unchanged)
│
└── rl_system/
    ├── config.py               ← ALL tunable parameters live here
    ├── database.py             ← SQLite persistence
    ├── notifier.py             ← Terminal + Windows toast + Discord
    ├── position_tracker.py     ← Position lifecycle + hard risk rules
    ├── rl_agent.py             ← Contextual bandit decision layer
    ├── run.py                  ← Main 60-second loop
    └── requirements_rl.txt     ← Additional pip requirements
```

## Setup

```bash
# 1. Install RL system dependencies
pip install -r rl_system/requirements_rl.txt

# 2. Make sure your API keys are set
export TRADIER_API_KEY="your_key"
export FRED_API_KEY="your_key"

# 3. Run from your main folder (where options_scanner.py lives)
cd your_folder
python rl_system/run.py --paper
```

## Usage

```bash
python rl_system/run.py              # live mode
python rl_system/run.py --paper      # paper trading mode (recommended first)
python rl_system/run.py --debug      # verbose output
python rl_system/run.py --status     # show positions + performance, then exit
python rl_system/run.py --weights    # show agent weight summary, then exit
```

## How It Works

### Every 60 seconds:
1. Checks all open positions for hard exit rules (stop loss, target hit, DTE)
2. Agent scores each open position: HOLD or EXIT
3. Agent scores top scanner candidates: ENTER or WAIT
4. Notifies you only when action changes AND confidence is above threshold
5. Always notifies on stop hits and urgent exits regardless of threshold

### Every 5 minutes:
- Re-runs the full options scanner to refresh candidates

### When a position closes:
- Reward is computed (R-multiples minus penalties)
- Agent weights updated via online gradient descent
- Cooldown set on that ticker (24 hours by default)

## Configuration

Edit `rl_system/config.py` to tune:
- `MAX_CONCURRENT_POSITIONS` — default 3
- `COOLDOWN_HOURS` — default 24
- `NOTIFY_CONFIDENCE_THRESHOLD` — default 0.60
- `STOP_LOSS_PCT` — default 0.50 (50%)
- `PROFIT_TARGET_PCT` — default 1.00 (100% = 2x)
- `NOTIFY_DISCORD_WEBHOOK_URL` — set to enable Discord alerts

## Manually Recording a Position

If you execute a trade manually that the scanner recommended,
record it so the system tracks it:

```python
# In a Python shell from your main folder:
import sys; sys.path.insert(0, 'rl_system')
from position_tracker import PositionTracker
t = PositionTracker()
t.manual_override_open(
    ticker="MU",
    strategy="BULL_CALL_SPREAD",
    strike=460.0,
    expiration="2026-03-27",
    entry_price=10.35,
    contracts=1
)
```

## Manually Closing a Position

```python
from position_tracker import PositionTracker
t = PositionTracker()
t.manual_override_close(position_id=1, current_price=19.50, reason="MANUAL")
```

## Discord Notifications

Set `NOTIFY_DISCORD_WEBHOOK_URL` in `config.py` to your webhook URL.
Format: `https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN`

## Database

All data persists in `scanner_data.db` (SQLite).
Query it directly with any SQLite browser or:

```bash
sqlite3 scanner_data.db
> SELECT * FROM trade_journal ORDER BY timestamp DESC LIMIT 20;
> SELECT * FROM positions WHERE status = 'OPEN';
> SELECT ticker, realized_r, exit_reason FROM positions WHERE status = 'CLOSED';
```

## Architecture Notes

- Hard risk rules (stop loss, max positions, cooldown, drawdown) live in
  `position_tracker.py` and **cannot** be overridden by the agent
- The learning layer only affects ENTER/EXIT timing confidence
- Agent starts pure rule-based and gradually blends in learned weights
  after 10+ closed trades
- All decisions are logged to DB with reasons and feature values
- Weights are saved to DB on every close and every 10 ticks
- Full state is restored on restart — no data lost between sessions
