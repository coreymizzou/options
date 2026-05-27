#!/usr/bin/env python3
"""Train the supervised entry model from labeled candidate_evaluations rows."""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ml_entry_model import FEATURE_NAMES, DEFAULT_MODEL_PATH, extract_candidate_features


def _json(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _target(row):
    for key in ("outcome_1d_r", "outcome_eod_r", "outcome_1h_r"):
        val = row[key]
        if val is not None:
            return float(val)
    return None


def load_training_rows(db_path: Path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT *
        FROM candidate_evaluations
        WHERE raw_scanner_data IS NOT NULL
          AND entry_price IS NOT NULL
          AND action IN ('PREFILTER_PASS', 'PREFILTER_REJECT')
          AND (
            outcome_1h_r IS NOT NULL OR
            outcome_eod_r IS NOT NULL OR
            outcome_1d_r IS NOT NULL
          )
        ORDER BY timestamp
        """
    ).fetchall()
    con.close()
    return rows


def build_matrix(rows):
    X, y = [], []
    for row in rows:
        scanner_result = _json(row["raw_scanner_data"])
        snapshot = _json(row["market_snapshot"])
        target = _target(row)
        if not scanner_result or target is None:
            continue
        features, _ = extract_candidate_features(
            scanner_result,
            snapshot,
            timestamp=row["timestamp"],
        )
        X.append(features)
        y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train entry model from candidate outcomes")
    parser.add_argument("--db", default="scanner_data_live.db", help="SQLite DB path")
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH), help="Model output path")
    parser.add_argument("--min-rows", type=int, default=100, help="Minimum labeled rows required")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    out_path = Path(args.out).resolve()
    rows = load_training_rows(db_path)
    X, y = build_matrix(rows)

    if len(y) < args.min_rows:
        print(f"Not enough labeled rows: {len(y)} found, need {args.min_rows}.")
        print("Run label_candidate_outcomes.py over multiple sessions first.")
        return 1

    try:
        import joblib
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"Missing ML dependency: {e}")
        print("Install with: pip install -r requirements.txt")
        return 1

    test_size = 0.25 if len(y) >= 200 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0
    directional_accuracy = float(np.mean((pred > 0) == (y_test > 0)))

    importance = []
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42)
        ranked = np.argsort(perm.importances_mean)[::-1][:12]
        importance = [
            (FEATURE_NAMES[i], float(perm.importances_mean[i]))
            for i in ranked
        ]
    except Exception:
        importance = []

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now().isoformat(),
        "n_samples": int(len(y)),
        "target": "coalesce(outcome_1d_r, outcome_eod_r, outcome_1h_r)",
        "metrics": {
            "mae": mae,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
            "test_rows": int(len(y_test)),
            "mean_target_r": float(np.mean(y)),
            "median_target_r": float(np.median(y)),
        },
        "feature_importance": importance,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)

    print(f"Saved entry model: {out_path}")
    print(f"Rows: {len(y)}  MAE: {mae:.3f}R  R2: {r2:.3f}  Directional accuracy: {directional_accuracy:.1%}")
    if importance:
        print("Top features:")
        for name, score in importance:
            print(f"  {name:<28} {score:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
