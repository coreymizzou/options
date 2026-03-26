"""
=============================================================================
notifier.py — Alert & Notification Layer
=============================================================================
Handles all user-facing alerts:
  - Terminal output (always on)
  - Windows toast notifications (optional)
  - Discord webhook (optional, add URL to config to enable)

Designed so new channels (SMS, email, Slack) can be added later
without changing any other module.
=============================================================================
"""

import json
import logging
import requests as req
from datetime import datetime
from typing import Optional

from config import (
    NOTIFY_TERMINAL,
    NOTIFY_WINDOWS_TOAST,
    NOTIFY_DISCORD_WEBHOOK_URL,
    DEBUG_MODE,
)

logger = logging.getLogger(__name__)

# ─── Try importing Windows toast library ─────────────────────────────────────
# Using 'winotify' — more stable than win10toast on modern Windows + Python 3.10+
# Install: pip install winotify
_TOAST_AVAILABLE = False
try:
    from winotify import Notification, audio
    _TOAST_AVAILABLE = True
except ImportError:
    try:
        # Fallback: plyer also works cross-platform
        from plyer import notification as plyer_notification
        _TOAST_AVAILABLE = True
        _USING_PLYER = True
    except ImportError:
        _USING_PLYER = False

_USING_PLYER = False  # reset — set properly below
try:
    from winotify import Notification, audio
    _TOAST_AVAILABLE = True
    _USING_PLYER = False
except ImportError:
    try:
        from plyer import notification as plyer_notification
        _TOAST_AVAILABLE = True
        _USING_PLYER = True
    except ImportError:
        pass


# =============================================================================
#  ACTION COLOR / PRIORITY MAPPING
# =============================================================================

ACTION_EMOJI = {
    "ENTER":    "🟢",
    "EXIT":     "🔴",
    "HOLD":     "🟡",
    "WAIT":     "⏳",
    "ALERT":    "🚨",
    "STOP_HIT": "🛑",
    "INFO":     "ℹ️",
}

ACTION_PRIORITY = {
    "STOP_HIT": 5,   # always notify regardless of threshold
    "EXIT":     4,
    "ENTER":    3,
    "HOLD":     2,
    "WAIT":     1,
    "INFO":     0,
}


# =============================================================================
#  ALERT DATA CLASS
# =============================================================================

class Alert:
    """
    Represents a single notification event.
    Passed to send() which routes it to all enabled channels.
    """
    def __init__(
        self,
        action: str,
        ticker: str,
        confidence: float,
        reasons: list,
        details: dict = None,
        force: bool = False,       # bypass confidence threshold
        priority: int = None,
    ):
        self.action     = action.upper()
        self.ticker     = ticker
        self.confidence = confidence
        self.reasons    = reasons or []
        self.details    = details or {}
        self.force      = force
        self.priority   = priority if priority is not None else ACTION_PRIORITY.get(self.action, 0)
        self.timestamp  = datetime.now()
        self.emoji      = ACTION_EMOJI.get(self.action, "•")

    def title(self) -> str:
        pct = f"{self.confidence * 100:.0f}%"
        return f"{self.emoji} {self.action} {self.ticker}  [{pct} confidence]"

    def body(self) -> str:
        lines = []
        if self.reasons:
            lines.append("Reasons:")
            for r in self.reasons[:4]:   # cap at 4 for toast readability
                lines.append(f"  • {r}")
        if self.details.get("entry"):
            lines.append(f"Entry: ${self.details['entry']}")
        if self.details.get("stop"):
            lines.append(f"Stop:  ${self.details['stop']}")
        if self.details.get("target"):
            lines.append(f"Target: ${self.details['target']}")
        return "\n".join(lines) if lines else self.action

    def terminal_str(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        conf_str = f"{self.confidence * 100:.0f}%"
        sep = "─" * 60
        lines = [
            sep,
            f"[{ts}]  {self.emoji}  {self.action}  {self.ticker}  "
            f"(confidence: {conf_str})",
        ]
        for r in self.reasons:
            lines.append(f"         • {r}")
        if self.details:
            for k, v in self.details.items():
                if v is not None:
                    lines.append(f"         {k}: {v}")
        lines.append(sep)
        return "\n".join(lines)


# =============================================================================
#  CHANNEL SENDERS
# =============================================================================

def _send_terminal(alert: Alert):
    """Print alert to terminal with formatting."""
    try:
        print(alert.terminal_str())
    except Exception as e:
        logger.error(f"Terminal notify failed: {e}")


def _send_windows_toast(alert: Alert):
    """Send Windows desktop toast notification."""
    if not _TOAST_AVAILABLE:
        if DEBUG_MODE:
            logger.debug("Windows toast not available — install winotify or plyer")
        return

    try:
        if not _USING_PLYER:
            # winotify
            toast = Notification(
                app_id="Options Scanner",
                title=alert.title(),
                msg=alert.body()[:256],   # toast body has char limit
                duration="short"
            )
            if alert.action in ("EXIT", "STOP_HIT", "ALERT"):
                toast.set_audio(audio.Default, loop=False)
            toast.show()
        else:
            # plyer fallback
            plyer_notification.notify(
                title=alert.title(),
                message=alert.body()[:256],
                app_name="Options Scanner",
                timeout=8
            )
    except Exception as e:
        logger.warning(f"Windows toast failed: {e}")


def _send_discord(alert: Alert):
    """Send alert to Discord webhook."""
    url = NOTIFY_DISCORD_WEBHOOK_URL
    if not url:
        return

    try:
        color_map = {
            "ENTER":    0x00FF00,   # green
            "EXIT":     0xFF4444,   # red
            "HOLD":     0xFFAA00,   # orange
            "WAIT":     0x888888,   # grey
            "STOP_HIT": 0xFF0000,   # bright red
        }
        color = color_map.get(alert.action, 0xAAAAAA)
        conf_str = f"{alert.confidence * 100:.0f}%"

        fields = []
        for r in alert.reasons[:5]:
            fields.append({"name": "•", "value": r, "inline": False})

        if alert.details.get("entry"):
            fields.append({"name": "Entry", "value": f"${alert.details['entry']}", "inline": True})
        if alert.details.get("stop"):
            fields.append({"name": "Stop",  "value": f"${alert.details['stop']}",  "inline": True})
        if alert.details.get("target"):
            fields.append({"name": "Target","value": f"${alert.details['target']}","inline": True})

        payload = {
            "embeds": [{
                "title":       alert.title(),
                "color":       color,
                "fields":      fields,
                "footer":      {"text": f"Options Scanner  •  {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"},
                "description": f"Confidence: {conf_str}"
            }]
        }
        resp = req.post(url, json=payload, timeout=5)
        if resp.status_code not in (200, 204):
            logger.warning(f"Discord webhook returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Discord notify failed: {e}")


# =============================================================================
#  PUBLIC SEND FUNCTION
# =============================================================================

def send(alert: Alert):
    """
    Route an Alert to all enabled notification channels.
    This is the single entry point for all notifications.

    To add a new channel in future (e.g. SMS, Slack):
      1. Write _send_sms(alert) or _send_slack(alert) above
      2. Add it to the channel list below
      3. Add its config toggle to config.py
    """
    channels = []

    if NOTIFY_TERMINAL:
        channels.append(_send_terminal)

    if NOTIFY_WINDOWS_TOAST and _TOAST_AVAILABLE:
        channels.append(_send_windows_toast)

    if NOTIFY_DISCORD_WEBHOOK_URL:
        channels.append(_send_discord)

    for channel_fn in channels:
        try:
            channel_fn(alert)
        except Exception as e:
            logger.error(f"Notification channel {channel_fn.__name__} failed: {e}")

    logger.info(
        f"Alert sent: {alert.action} {alert.ticker} "
        f"conf={alert.confidence:.2f} force={alert.force}"
    )


def notify_entry(ticker: str, confidence: float, reasons: list,
                 entry: float = None, stop: float = None,
                 target: float = None, strategy: str = None,
                 strike: float = None, expiration: str = None,
                 contracts: int = None, trade_summary: str = None):
    """Convenience wrapper for ENTER alerts."""
    details = {}
    if strategy:       details["strategy"]       = strategy
    if trade_summary:  details["trade"]          = trade_summary
    elif strike:       details["strike"]         = f"${strike}"
    if expiration and not trade_summary:
                       details["expiration"]     = expiration
    if entry:          details["entry"]          = f"${entry}"
    if stop:           details["stop"]           = f"${stop}"
    if target:         details["target"]         = f"${target}"
    if contracts:      details["contracts"]      = contracts

    alert = Alert(
        action="ENTER",
        ticker=ticker,
        confidence=confidence,
        reasons=reasons,
        details=details
    )
    send(alert)


def notify_exit(ticker: str, confidence: float, reasons: list,
                unrealized_pnl: float = None, exit_price: float = None,
                force: bool = False):
    """Convenience wrapper for EXIT alerts."""
    details = {}
    if unrealized_pnl is not None:
        sign = "+" if unrealized_pnl >= 0 else ""
        details["unrealized_pnl"] = f"{sign}${unrealized_pnl:.2f}"
    if exit_price:
        details["exit_price"] = exit_price

    alert = Alert(
        action="EXIT",
        ticker=ticker,
        confidence=confidence,
        reasons=reasons,
        details=details,
        force=force
    )
    send(alert)


def notify_stop_hit(ticker: str, loss: float, position_id: int = None):
    """Convenience wrapper for forced stop-loss exits."""
    reasons = [
        f"Position down ${abs(loss):.2f} — stop loss triggered",
        "Hard risk rule: close immediately"
    ]
    details = {"loss": f"-${abs(loss):.2f}"}
    if position_id:
        details["position_id"] = position_id

    alert = Alert(
        action="STOP_HIT",
        ticker=ticker,
        confidence=1.0,       # always 100% — hard rule
        reasons=reasons,
        details=details,
        force=True            # always bypass confidence threshold
    )
    send(alert)


def notify_info(message: str, ticker: str = "SYSTEM"):
    """Send a plain informational message."""
    alert = Alert(
        action="INFO",
        ticker=ticker,
        confidence=1.0,
        reasons=[message]
    )
    send(alert)
