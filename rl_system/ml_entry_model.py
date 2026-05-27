"""
Supervised entry model helpers.

This module is intentionally optional: live trading falls back to the existing
agent if no trained model artifact exists. The model is trained offline from
candidate_evaluations rows after outcome labeling.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "entry_model.pkl"

FEATURE_NAMES = [
    "confluence_score",
    "entry_price",
    "ivr",
    "current_iv",
    "rsi",
    "intraday_rsi",
    "rel_volume",
    "above_vwap",
    "intraday_above_vwap",
    "spy_change_pct",
    "qqq_change_pct",
    "flow_count",
    "top_flow_score",
    "top_flow_premium_log",
    "top_flow_high_conf",
    "strike_distance_pct",
    "dte",
    "days_to_earnings",
    "regime_trending_up",
    "regime_trending_down",
    "regime_risk_off",
    "direction_bullish",
    "direction_bearish",
    "strategy_long_call",
    "strategy_long_put",
    "strategy_bull_call_spread",
    "strategy_bear_put_spread",
    "strategy_long_straddle",
    "broad_market_bullish",
    "broad_market_bearish",
    "broad_market_mixed",
]

_MODEL_CACHE: Optional[Dict[str, Any]] = None
_MODEL_MTIME: Optional[float] = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_json(value: Any) -> Dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _days_to_expiration(expiration: Any, timestamp: Any = None) -> float:
    if not expiration:
        return 0.0
    try:
        from datetime import datetime
        start = datetime.fromisoformat(str(timestamp)) if timestamp else datetime.now()
        end = datetime.strptime(str(expiration)[:10], "%Y-%m-%d")
        return max((end - start).days, 0)
    except Exception:
        return 0.0


def extract_candidate_features(
    scanner_result: Dict,
    market_snapshot: Optional[Dict] = None,
    timestamp: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Convert a scanner candidate into the tabular feature vector."""
    market_snapshot = market_snapshot or {}
    trade = scanner_result.get("trade", {}) or {}
    tech = scanner_result.get("tech", {}) or {}
    vol = scanner_result.get("vol", {}) or {}
    conf = scanner_result.get("confluence", {}) or {}
    sector = scanner_result.get("sector", {}) or {}
    flow = scanner_result.get("flow", []) or []
    pricing = scanner_result.get("pricing", {}) or {}
    main_leg = trade.get("main_leg", {}) or {}

    direction = str(trade.get("direction") or "").upper()
    strategy = str(trade.get("strategy") or "").upper()
    regime = str(
        market_snapshot.get("regime") or scanner_result.get("regime") or trade.get("regime") or ""
    ).upper()
    broad_market = str(sector.get("broad_market") or market_snapshot.get("broad_market") or "").upper()

    spot = _safe_float(scanner_result.get("spot") or trade.get("spot"), 0.0)
    strike = _safe_float(main_leg.get("strike"), 0.0)
    strike_distance_pct = abs(strike - spot) / spot if spot > 0 and strike > 0 else 0.0
    top_flow = flow[0] if flow else {}
    premium = _safe_float(top_flow.get("premium_paid"), 0.0)

    values = {
        "confluence_score": _safe_float(conf.get("score"), 0.0),
        "entry_price": _safe_float(pricing.get("entry"), 0.0),
        "ivr": _safe_float(vol.get("ivr"), 50.0),
        "current_iv": _safe_float(vol.get("current_iv"), 0.0),
        "rsi": _safe_float(tech.get("rsi") or market_snapshot.get("rsi"), 50.0),
        "intraday_rsi": _safe_float(tech.get("intraday_rsi"), 50.0),
        "rel_volume": _safe_float(tech.get("rel_volume"), 1.0),
        "above_vwap": 1.0 if tech.get("above_vwap") or market_snapshot.get("above_vwap") else 0.0,
        "intraday_above_vwap": 1.0 if tech.get("intraday_above_vwap") else 0.0,
        "spy_change_pct": _safe_float(sector.get("spy_change_pct") or market_snapshot.get("spy_change_pct"), 0.0),
        "qqq_change_pct": _safe_float(sector.get("qqq_change_pct"), 0.0),
        "flow_count": float(len(flow)),
        "top_flow_score": _safe_float(top_flow.get("score"), 0.0),
        "top_flow_premium_log": math.log1p(max(premium, 0.0)),
        "top_flow_high_conf": 1.0 if top_flow.get("dir_confidence") == "HIGH" else 0.0,
        "strike_distance_pct": strike_distance_pct,
        "dte": _days_to_expiration(main_leg.get("exp") or trade.get("exp"), timestamp),
        "days_to_earnings": _safe_float(scanner_result.get("days_to_earnings"), 60.0),
        "regime_trending_up": 1.0 if regime == "TRENDING_UP" else 0.0,
        "regime_trending_down": 1.0 if regime == "TRENDING_DOWN" else 0.0,
        "regime_risk_off": 1.0 if regime == "RISK_OFF" else 0.0,
        "direction_bullish": 1.0 if direction == "BULLISH" else 0.0,
        "direction_bearish": 1.0 if direction == "BEARISH" else 0.0,
        "strategy_long_call": 1.0 if strategy == "LONG_CALL" else 0.0,
        "strategy_long_put": 1.0 if strategy == "LONG_PUT" else 0.0,
        "strategy_bull_call_spread": 1.0 if strategy == "BULL_CALL_SPREAD" else 0.0,
        "strategy_bear_put_spread": 1.0 if strategy == "BEAR_PUT_SPREAD" else 0.0,
        "strategy_long_straddle": 1.0 if strategy == "LONG_STRADDLE" else 0.0,
        "broad_market_bullish": 1.0 if broad_market == "BULLISH" else 0.0,
        "broad_market_bearish": 1.0 if broad_market == "BEARISH" else 0.0,
        "broad_market_mixed": 1.0 if broad_market == "MIXED" else 0.0,
    }
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32), values


def model_path() -> Path:
    configured = getattr(cfg, "ML_ENTRY_MODEL_PATH", "")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(__file__).resolve().parent.parent / path
    return DEFAULT_MODEL_PATH


def load_model(force: bool = False) -> Optional[Dict[str, Any]]:
    """Load the trained model artifact, caching by mtime."""
    global _MODEL_CACHE, _MODEL_MTIME
    path = model_path()
    if not path.exists():
        _MODEL_CACHE = None
        _MODEL_MTIME = None
        return None
    mtime = path.stat().st_mtime
    if not force and _MODEL_CACHE is not None and _MODEL_MTIME == mtime:
        return _MODEL_CACHE
    try:
        import joblib
        artifact = joblib.load(path)
        if artifact.get("feature_names") != FEATURE_NAMES:
            logger.warning("ML entry model feature mismatch; ignoring artifact")
            return None
        _MODEL_CACHE = artifact
        _MODEL_MTIME = mtime
        logger.info(
            "Loaded ML entry model from %s (n=%s)",
            path,
            artifact.get("n_samples"),
        )
        return artifact
    except Exception as e:
        logger.warning("Failed to load ML entry model: %s", e)
        return None


def score_candidate(scanner_result: Dict, market_snapshot: Dict) -> Optional[Dict[str, Any]]:
    """Return model prediction for a live candidate, or None if unavailable."""
    if not getattr(cfg, "ML_ENTRY_MODEL_ENABLED", True):
        return None
    artifact = load_model()
    if not artifact:
        return None
    if int(artifact.get("n_samples") or 0) < getattr(cfg, "ML_MIN_TRAINING_ROWS", 100):
        return None
    try:
        features, feature_map = extract_candidate_features(
            scanner_result,
            market_snapshot,
            timestamp=scanner_result.get("timestamp"),
        )
        expected_r = float(artifact["model"].predict([features])[0])
        confidence = float(1.0 / (1.0 + math.exp(-expected_r * 3.0)))
        return {
            "expected_r": expected_r,
            "confidence": confidence,
            "model_version": artifact.get("trained_at"),
            "n_samples": artifact.get("n_samples"),
            "features": feature_map,
        }
    except Exception as e:
        logger.warning("ML entry score failed: %s", e)
        return None
