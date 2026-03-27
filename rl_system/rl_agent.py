"""
=============================================================================
rl_agent.py — Contextual Bandit / Hybrid Decision Layer
=============================================================================
Implements a lightweight online-learning decision agent.

Architecture:
  - Starts from rule-based priors (the scanner's existing logic)
  - Uses a linear contextual bandit model per action type
  - Updates weights online after each position closes (reward observed)
  - Outputs action + confidence + interpretable reasons every tick
  - NEVER overrides hard risk rules (those live in position_tracker.py)

Actions:
  ENTER  — recommend opening this position now
  HOLD   — keep existing position, no action
  EXIT   — recommend closing this position now
  WAIT   — scanner signal present but conditions not ideal, wait

Reward signal (R-multiples):
  reward = (realized_pnl / initial_risk)
           - stop_penalty (if stop hit)
           - drawdown_penalty (if rolling drawdown breached)
           - churn_penalty (if held < min ticks)
=============================================================================
"""

import json
import math
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

import database as db
from config import (
    FEATURE_NAMES,
    AGENT_LEARNING_RATE,
    AGENT_EXPLORATION_RATE,
    AGENT_MIN_SAMPLES_TO_LEARN,
    AGENT_WEIGHT_DECAY,
    AGENT_SAVE_INTERVAL_TICKS,
    REWARD_STOP_PENALTY,
    REWARD_DRAWDOWN_PENALTY,
    REWARD_CHURN_PENALTY,
    REWARD_MIN_HOLD_TICKS,
    MAX_ROLLING_DRAWDOWN_PCT,
    ACCOUNT_SIZE,
    ENTER_CONFIDENCE_THRESHOLD,
    EXIT_CONFIDENCE_THRESHOLD,
    HOLD_IS_DEFAULT,
    DEBUG_MODE,
)

logger = logging.getLogger(__name__)

N_FEATURES = len(FEATURE_NAMES)


# =============================================================================
#  FEATURE EXTRACTION
# =============================================================================

def extract_features(
    position: Optional[Dict],
    scanner_result: Optional[Dict],
    market_snapshot: Dict,
    ticks_held: int = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Build normalized feature vector from position + market state.
    Returns (feature_array, human_readable_feature_dict)

    Features are designed to be:
      - Bounded [0, 1] or [-1, 1] for numerical stability
      - Interpretable — each has a clear meaning
      - Available from existing data feeds
    """
    feat = np.zeros(N_FEATURES, dtype=np.float32)
    desc = {}

    # ── Helpers
    def safe(val, default=0.0):
        try:
            return float(val) if val is not None else default
        except Exception:
            return default

    # 0. unrealized_r — current P&L in R-multiples (clipped to [-2, 2])
    ur = safe(market_snapshot.get("unrealized_r"), 0.0)
    feat[0] = np.clip(ur / 2.0, -1.0, 1.0)
    desc["unrealized_r"] = round(ur, 3)

    # 1. dte_fraction — how much time is left relative to entry DTE
    dte_remaining = safe(market_snapshot.get("dte_remaining"), 30)
    entry_dte     = safe(position.get("entry_dte", 30) if position else 30, 30)
    dte_frac      = dte_remaining / max(entry_dte, 1)
    feat[1] = np.clip(dte_frac, 0.0, 1.0)
    desc["dte_fraction"] = round(dte_frac, 3)

    # 2. theta_decay_fraction — how much premium has theta eaten
    entry_cost    = safe(position.get("entry_cost", 1) if position else 1, 1)
    theta_today   = safe(market_snapshot.get("theta_today"), 0.0)
    days_held     = safe(market_snapshot.get("days_since_entry"), 0)
    theta_total   = abs(theta_today) * days_held
    theta_frac    = min(theta_total / max(entry_cost, 1), 1.0)
    feat[2] = theta_frac
    desc["theta_decay_fraction"] = round(theta_frac, 3)

    # 3. iv_rank_normalized
    ivr = safe(market_snapshot.get("ivr_current"), 50.0)
    feat[3] = ivr / 100.0
    desc["iv_rank_normalized"] = round(ivr / 100.0, 3)

    # 4. spy_trend — +1 bull, 0 neutral, -1 bear
    spy_chg = safe(market_snapshot.get("spy_change_pct"), 0.0)
    spy_val = 1.0 if spy_chg > 0.2 else (-1.0 if spy_chg < -0.2 else 0.0)
    feat[4] = spy_val
    desc["spy_trend"] = spy_val

    # 5. rsi_normalized
    rsi = safe(market_snapshot.get("rsi"), 50.0)
    feat[5] = rsi / 100.0
    desc["rsi_normalized"] = round(rsi / 100.0, 3)

    # 6. flow_score_normalized
    flow_score = safe(market_snapshot.get("flow_score"), 0)
    feat[6] = min(flow_score / 15.0, 1.0)
    desc["flow_score_normalized"] = round(feat[6], 3)

    # 7. above_vwap
    above_vwap = 1.0 if market_snapshot.get("above_vwap") else 0.0
    feat[7] = above_vwap
    desc["above_vwap"] = int(above_vwap)

    # 8. regime_score — 1 trending, 0 ranging, -1 risk_off
    regime = market_snapshot.get("regime", "RANGING")
    regime_val = (
        1.0  if regime in ("TRENDING_UP", "TRENDING_DOWN") else
        -1.0 if regime == "RISK_OFF" else
        0.0
    )
    feat[8] = regime_val
    desc["regime_score"] = regime_val

    # 9. ticks_held_normalized
    feat[9] = min(ticks_held / 100.0, 1.0)
    desc["ticks_held_normalized"] = round(feat[9], 3)

    # 10. spread_vs_target — how far toward profit target we are
    if position:
        entry_p  = safe(position.get("entry_price"), 0)
        target_p = safe(position.get("target_price"), entry_p * 2)
        opt_mid  = safe(market_snapshot.get("option_mid"), entry_p)
        rng = target_p - entry_p
        svt = (opt_mid - entry_p) / max(rng, 0.01)
        feat[10] = np.clip(svt, -1.0, 2.0) / 2.0   # normalize to [-0.5, 1]
        desc["spread_vs_target"] = round(svt, 3)
    else:
        feat[10] = 0.0
        desc["spread_vs_target"] = 0.0

    # 11. days_since_entry_norm
    feat[11] = min(days_held / 30.0, 1.0)
    desc["days_since_entry_norm"] = round(feat[11], 3)

    return feat, desc


# =============================================================================
#  LINEAR WEIGHT MODEL
# =============================================================================

class LinearBandit:
    """
    A simple online-learning linear model for one action type.
    score = dot(weights, features) + bias
    Updated via stochastic gradient descent on observed rewards.

    Interpretable by design — weights[i] tells you how much feature i
    contributes to the score for this action.
    """

    def __init__(self, label: str, n_features: int = N_FEATURES):
        self.label      = label
        self.n_features = n_features
        self.weights    = np.zeros(n_features, dtype=np.float32)
        self.bias       = 0.0
        self.n_updates  = 0
        self.mean_reward = 0.0
        self._reward_history: List[float] = []

        # Try to restore from DB
        self._load()

    def score(self, features: np.ndarray) -> float:
        """Raw dot-product score before sigmoid."""
        return float(np.dot(self.weights, features) + self.bias)

    def confidence(self, features: np.ndarray) -> float:
        """Sigmoid-transformed score → probability-like confidence in [0, 1]."""
        s = self.score(features)
        return float(1.0 / (1.0 + math.exp(-s)))

    def update(self, features: np.ndarray, reward: float):
        """
        Online SGD update.
        reward: R-multiple outcome (positive = good, negative = bad)
        """
        lr      = AGENT_LEARNING_RATE
        wd      = AGENT_WEIGHT_DECAY
        pred    = self.score(features)
        error   = reward - pred

        # Gradient step
        self.weights += lr * error * features
        self.bias    += lr * error

        # L2 weight decay
        self.weights *= (1 - wd)

        self.n_updates += 1
        self._reward_history.append(reward)
        if len(self._reward_history) > 200:
            self._reward_history = self._reward_history[-200:]
        self.mean_reward = float(np.mean(self._reward_history))

    def top_features(self, features: np.ndarray, n: int = 3) -> List[str]:
        """Return names of the n features most contributing to current score."""
        contributions = self.weights * features
        top_idx = np.argsort(np.abs(contributions))[::-1][:n]
        reasons = []
        for i in top_idx:
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
            contrib = contributions[i]
            direction = "↑" if contrib > 0 else "↓"
            reasons.append(
                f"{name}: {FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else name} "
                f"{direction} {abs(contrib):.3f}"
            )
        return reasons

    def _load(self):
        """Restore weights from database if available."""
        saved = db.load_agent_weights(self.label)
        if saved:
            w = saved.get("weights", [])
            if len(w) == self.n_features:
                self.weights    = np.array(w, dtype=np.float32)
                self.bias       = float(saved.get("bias", 0.0))
                self.n_updates  = int(saved.get("n_updates", 0))
                self.mean_reward = float(saved.get("mean_reward", 0.0))
                logger.info(
                    f"Agent weights loaded: {self.label} "
                    f"({self.n_updates} updates, mean_R={self.mean_reward:.3f})"
                )
            else:
                logger.warning(
                    f"Saved weights for {self.label} have wrong shape "
                    f"({len(w)} vs {self.n_features}) — starting fresh"
                )

    def save(self):
        """Persist weights to database."""
        db.save_agent_weights(
            label       = self.label,
            weights     = self.weights.tolist(),
            bias        = self.bias,
            n_updates   = self.n_updates,
            mean_reward = self.mean_reward
        )


# =============================================================================
#  RULE-BASED PRIORS
# =============================================================================

def rule_based_enter_score(scanner_result: Dict, market_snapshot: Dict) -> Tuple[float, List[str]]:
    """
    Rule-based prior for ENTER decisions.
    This is the scanner's existing logic translated into a confidence score.
    Used as the baseline before the learning layer has enough data.
    """
    score   = 0.0
    reasons = []
    conf    = scanner_result.get("confluence", {})
    trade   = scanner_result.get("trade", {})
    vol     = scanner_result.get("vol", {})
    tech    = scanner_result.get("tech", {})

    # Confluence score → base confidence
    conf_score = conf.get("score", 0)
    if conf_score >= 10:
        score += 0.40
        reasons.append(f"A+ confluence ({conf_score}pts)")
    elif conf_score >= 7:
        score += 0.25
        reasons.append(f"Strong confluence ({conf_score}pts)")
    elif conf_score >= 5:
        score += 0.10
        reasons.append(f"Moderate confluence ({conf_score}pts)")
    else:
        score -= 0.10
        reasons.append(f"Weak confluence ({conf_score}pts) — below threshold")

    # Flow signal
    flow = scanner_result.get("flow", [])
    if flow:
        top = flow[0]
        if top.get("dir_confidence") == "HIGH":
            score += 0.20
            reasons.append(
                f"High-confidence flow: {top.get('aggressor')} "
                f"${top.get('premium_paid', 0)/1000:.0f}k"
            )
        else:
            score += 0.10
            reasons.append("Ambiguous flow signal")

    # IV environment fit
    ivr = vol.get("ivr", 50) or 50
    strategy = trade.get("strategy", "")
    if ("SPREAD" in strategy and ivr > 50) or ("LONG" in strategy and ivr < 35):
        score += 0.15
        reasons.append(f"IV environment matches strategy (IVR {ivr:.0f})")

    # Market direction alignment
    spy_chg = market_snapshot.get("spy_change_pct", 0) or 0
    direction = trade.get("direction", "NEUTRAL")
    if (direction == "BULLISH" and spy_chg > 0) or (direction == "BEARISH" and spy_chg < 0):
        score += 0.10
        reasons.append(f"Market direction aligned (SPY {spy_chg:+.2f}%)")
    elif (direction == "BULLISH" and spy_chg < -0.5) or (direction == "BEARISH" and spy_chg > 0.5):
        score -= 0.15
        reasons.append(f"Going against market (SPY {spy_chg:+.2f}%)")

    return min(max(score, 0.0), 1.0), reasons


def rule_based_exit_score(position: Dict, market_snapshot: Dict,
                           ticks_held: int) -> Tuple[float, List[str]]:
    """
    Rule-based prior for EXIT decisions.
    Conservative — default to HOLD unless a clear reason to exit.

    Design principles:
    - No single factor should push score above threshold alone
    - Need multiple confirming signals to recommend EXIT
    - P&L-based signals only fire with meaningful moves (not noise)
    - DTE only matters in final stretch (< 7 days = hard rule anyway)
    - If no live price data available, score stays near 0
    """
    score   = 0.0
    reasons = []

    unrealized_r = market_snapshot.get("unrealized_r", 0) or 0
    dte_rem      = market_snapshot.get("dte_remaining") or 30
    entry_dte    = position.get("entry_dte") or 30

    # ── Only score P&L signals if we have a meaningful price move
    # If unrealized_r is exactly 0.0 it almost certainly means no live
    # price data — don't treat flat as a signal either way
    has_live_price = (
        market_snapshot.get("option_mid") is not None and
        market_snapshot.get("option_mid") != position.get("entry_price")
    )

    if has_live_price:
        # Strong profit — lean toward taking it (but not aggressively)
        if unrealized_r >= 0.90:
            score += 0.35
            reasons.append(f"Unrealized gain {unrealized_r:.2f}R — near target")
        elif unrealized_r >= 0.60:
            score += 0.20
            reasons.append(f"Unrealized gain {unrealized_r:.2f}R")

        # Significant loss building
        if unrealized_r <= -0.35:
            score += 0.25
            reasons.append(f"Position declining: {unrealized_r:.2f}R")
        elif unrealized_r <= -0.25:
            score += 0.10
            reasons.append(f"Loss building: {unrealized_r:.2f}R")
    else:
        reasons.append("No live price data — holding position")

    # ── DTE urgency — only in final 7 days (before hard rule fires)
    # Hard rule closes at CLOSE_BEFORE_DTE (7 days) so only score
    # the 7-14 day window as a soft warning
    if dte_rem is not None and dte_rem <= 14 and dte_rem > 7:
        score += 0.15
        reasons.append(f"Only {dte_rem} DTE — approaching force-close threshold")

    # ── Market reversal against position — needs strong move to matter
    spy_chg   = market_snapshot.get("spy_change_pct", 0) or 0
    direction = position.get("direction", "NEUTRAL")
    if (direction == "BULLISH" and spy_chg < -1.5) or \
       (direction == "BEARISH" and spy_chg > 1.5):
        score += 0.20
        reasons.append(f"Strong market reversal against position (SPY {spy_chg:+.2f}%)")

    return min(max(score, 0.0), 1.0), reasons


# =============================================================================
#  REWARD COMPUTATION
# =============================================================================

def compute_reward(realized_r: float, exit_reason: str,
                   ticks_held: int, rolling_drawdown: float) -> float:
    """
    Compute the R-multiple-based reward used to update agent weights.
    reward = base_r - stop_penalty - drawdown_penalty - churn_penalty
    """
    reward = realized_r

    # Stop penalty
    if "STOP" in exit_reason.upper():
        reward -= REWARD_STOP_PENALTY

    # Drawdown penalty
    if abs(rolling_drawdown) > MAX_ROLLING_DRAWDOWN_PCT:
        reward -= REWARD_DRAWDOWN_PENALTY

    # Churn penalty — penalize very short holds
    if ticks_held < REWARD_MIN_HOLD_TICKS and realized_r <= 0:
        reward -= REWARD_CHURN_PENALTY

    return reward


# =============================================================================
#  MAIN AGENT CLASS
# =============================================================================

class DecisionAgent:
    """
    Contextual bandit agent that scores ENTER / HOLD / EXIT / WAIT actions.

    Phase 1 behavior:
      - Uses rule-based priors when n_updates < AGENT_MIN_SAMPLES_TO_LEARN
      - Blends rules + learned weights as more data accumulates
      - Always logs reasoning for every decision
      - Saves weights to DB every AGENT_SAVE_INTERVAL_TICKS ticks
    """

    def __init__(self):
        self.enter_model = LinearBandit("enter_weights")
        self.exit_model  = LinearBandit("exit_weights")
        self._tick_count = 0
        self._exploration_rate = AGENT_EXPLORATION_RATE
        logger.info(
            f"DecisionAgent initialized — "
            f"enter_model: {self.enter_model.n_updates} updates, "
            f"exit_model: {self.exit_model.n_updates} updates"
        )

    # ─── Entry Decision ───────────────────────────────────────────────────────

    def score_entry(
        self,
        scanner_result: Dict,
        market_snapshot: Dict
    ) -> Tuple[str, float, List[str]]:
        """
        Score whether to ENTER or WAIT on a scanner recommendation.
        Returns (action, confidence, reasons)
        """
        features, feat_desc = extract_features(
            position=None,
            scanner_result=scanner_result,
            market_snapshot=market_snapshot,
            ticks_held=0
        )

        # Rule-based prior
        rule_score, rule_reasons = rule_based_enter_score(scanner_result, market_snapshot)

        # Learned model confidence
        learned_conf = self.enter_model.confidence(features)
        n_updates    = self.enter_model.n_updates

        # Blend: start pure rule-based, gradually shift to learned model
        if n_updates < AGENT_MIN_SAMPLES_TO_LEARN:
            blend_w = 0.0   # pure rules
        else:
            blend_w = min((n_updates - AGENT_MIN_SAMPLES_TO_LEARN) / 50.0, 0.60)

        final_conf = (1 - blend_w) * rule_score + blend_w * learned_conf

        # Exploration: occasionally randomize to gather diverse data
        if random.random() < self._exploration_rate and n_updates < 50:
            final_conf = random.uniform(0.3, 0.8)
            rule_reasons = [f"[EXPLORE] Random confidence for learning diversity"] + rule_reasons

        # Feature importance from learned model
        learned_reasons = []
        if n_updates >= AGENT_MIN_SAMPLES_TO_LEARN:
            learned_reasons = self.enter_model.top_features(features)

        reasons = rule_reasons + (["─── Learned factors ───"] + learned_reasons if learned_reasons else [])

        # Decision
        action = "ENTER" if final_conf >= ENTER_CONFIDENCE_THRESHOLD else "WAIT"

        self._log_decision(
            action=action, confidence=final_conf,
            reasons=reasons, feat_desc=feat_desc,
            ticker=scanner_result.get("ticker", "?"),
            is_entry=True
        )

        return action, final_conf, reasons

    # ─── Exit Decision ────────────────────────────────────────────────────────

    def score_exit(
        self,
        position: Dict,
        market_snapshot: Dict,
        ticks_held: int
    ) -> Tuple[str, float, List[str]]:
        """
        Score whether to EXIT or HOLD an open position.
        Returns (action, confidence, reasons)
        """
        features, feat_desc = extract_features(
            position=position,
            scanner_result=None,
            market_snapshot=market_snapshot,
            ticks_held=ticks_held
        )

        # Rule-based prior
        rule_score, rule_reasons = rule_based_exit_score(
            position, market_snapshot, ticks_held
        )

        # Learned model
        learned_conf = self.exit_model.confidence(features)
        n_updates    = self.exit_model.n_updates

        if n_updates < AGENT_MIN_SAMPLES_TO_LEARN:
            blend_w = 0.0
        else:
            blend_w = min((n_updates - AGENT_MIN_SAMPLES_TO_LEARN) / 50.0, 0.60)

        final_conf = (1 - blend_w) * rule_score + blend_w * learned_conf

        # Feature importance
        learned_reasons = []
        if n_updates >= AGENT_MIN_SAMPLES_TO_LEARN:
            learned_reasons = self.exit_model.top_features(features)

        reasons = rule_reasons + (["─── Learned factors ───"] + learned_reasons if learned_reasons else [])

        # HOLD is default — only exit if confidence clears threshold
        if HOLD_IS_DEFAULT:
            action = "EXIT" if final_conf >= EXIT_CONFIDENCE_THRESHOLD else "HOLD"
        else:
            # Non-default: exit unless confidence clearly favors holding
            action = "HOLD" if final_conf < (1 - EXIT_CONFIDENCE_THRESHOLD) else "EXIT"

        self._log_decision(
            action=action, confidence=final_conf,
            reasons=reasons, feat_desc=feat_desc,
            ticker=position.get("ticker", "?"),
            is_entry=False,
            position_id=position.get("id")
        )

        return action, final_conf, reasons

    # ─── Online Learning Update ───────────────────────────────────────────────

    def update_on_close(
        self,
        position: Dict,
        exit_reason: str,
        realized_r: float,
        entry_market_snapshot: Dict,
        exit_market_snapshot: Dict,
        ticks_held: int,
        rolling_drawdown: float
    ):
        """
        Update both models using the observed reward from a closed position.
        Called automatically when a position is closed.
        """
        reward = compute_reward(
            realized_r      = realized_r,
            exit_reason     = exit_reason,
            ticks_held      = ticks_held,
            rolling_drawdown = rolling_drawdown
        )

        # Update enter model — should have entered? (positive reward = yes)
        entry_features, _ = extract_features(
            position=None,
            scanner_result=None,
            market_snapshot=entry_market_snapshot,
            ticks_held=0
        )
        self.enter_model.update(entry_features, reward)

        # Update exit model — was the exit timing good?
        exit_features, _ = extract_features(
            position=position,
            scanner_result=None,
            market_snapshot=exit_market_snapshot,
            ticks_held=ticks_held
        )
        self.exit_model.update(exit_features, reward)

        logger.info(
            f"Agent updated: {position.get('ticker')} "
            f"realized_r={realized_r:.3f} reward={reward:.3f} "
            f"enter_updates={self.enter_model.n_updates} "
            f"exit_updates={self.exit_model.n_updates}"
        )

        db.log_journal_event(
            "AGENT_UPDATE",
            ticker=position.get("ticker"),
            position_id=position.get("id"),
            action="UPDATE",
            confidence=None,
            reason_summary=f"Agent updated: R={realized_r:.3f} → reward={reward:.3f}",
            details={
                "realized_r":     realized_r,
                "reward":         reward,
                "exit_reason":    exit_reason,
                "ticks_held":     ticks_held,
                "enter_updates":  self.enter_model.n_updates,
                "exit_updates":   self.exit_model.n_updates,
                "enter_mean_r":   round(self.enter_model.mean_reward, 3),
                "exit_mean_r":    round(self.exit_model.mean_reward, 3),
            }
        )

        # Periodic weight save
        self._tick_count += 1
        if self._tick_count % AGENT_SAVE_INTERVAL_TICKS == 0:
            self.save_weights()

    def save_weights(self):
        self.enter_model.save()
        self.exit_model.save()
        logger.debug("Agent weights saved to DB")

    # ─── Internal Logging ────────────────────────────────────────────────────

    def _log_decision(self, action: str, confidence: float, reasons: List[str],
                       feat_desc: Dict, ticker: str, is_entry: bool,
                       position_id: int = None):
        """Log every decision to DB for debugging and analysis."""
        if not DEBUG_MODE and action == "HOLD":
            return   # suppress HOLD spam in non-debug mode

        db.log_journal_event(
            event_type   = "ENTRY_REC" if is_entry else "EXIT_REC",
            ticker       = ticker,
            position_id  = position_id,
            action       = action,
            confidence   = confidence,
            reason_summary = f"{action} {ticker} conf={confidence:.2f}",
            details      = {
                "reasons":      reasons[:6],
                "features":     feat_desc,
                "is_entry":     is_entry,
            }
        )

    # ─── Diagnostics ─────────────────────────────────────────────────────────

    def print_weight_summary(self):
        """Print current weights in human-readable format."""
        print("\n─── Agent Weight Summary ───")
        for model_label, model in [("ENTER", self.enter_model), ("EXIT", self.exit_model)]:
            print(f"\n  [{model_label} model]  updates={model.n_updates}  "
                  f"mean_reward={model.mean_reward:.3f}  bias={model.bias:.3f}")
            sorted_idx = np.argsort(np.abs(model.weights))[::-1]
            for i in sorted_idx[:6]:
                name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
                print(f"    {name:<30} {model.weights[i]:+.4f}")
        print()
