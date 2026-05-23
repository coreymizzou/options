"""
trade_analytics.py — Outcome Analysis For Scanner And Agent Tuning

Reads closed trades from the SQLite database and groups realized R by the
features most likely to explain performance. This is intentionally lightweight:
it depends only on the existing database schema and raw_scanner_data snapshots.
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


DEFAULT_DB = Path("./scanner_data_live.db")


def _load_rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM positions
            WHERE status = 'CLOSED'
            ORDER BY exit_time DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _load_candidate_rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='candidate_evaluations'
            """
        ).fetchone()
        if not exists:
            return []
        rows = conn.execute(
            """
            SELECT *
            FROM candidate_evaluations
            ORDER BY timestamp DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _json_dict(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _entry_hour(row: dict) -> str:
    try:
        dt = datetime.fromisoformat(row.get("entry_time") or "")
        return f"{dt.hour:02d}:00"
    except Exception:
        return "unknown"


def _score_band(score) -> str:
    try:
        score = int(score)
    except Exception:
        return "unknown"
    if score >= 12:
        return "12+"
    if score >= 10:
        return "10-11"
    if score >= 8:
        return "8-9"
    return "<8"


def _first_flow_conf(raw: dict) -> str:
    flow = raw.get("flow") or []
    if flow and isinstance(flow[0], dict):
        return flow[0].get("dir_confidence") or "unknown"
    return "none"


def _regime(raw: dict) -> str:
    regime = raw.get("regime_data") or {}
    if isinstance(regime, dict):
        return regime.get("regime") or raw.get("trade", {}).get("regime") or "unknown"
    return raw.get("trade", {}).get("regime") or "unknown"


def _above_vwap(raw: dict) -> str:
    val = (raw.get("tech") or {}).get("above_vwap")
    if val is True:
        return "above"
    if val is False:
        return "below"
    return "unknown"


def _ivr_band(row: dict) -> str:
    ivr = row.get("entry_ivr")
    try:
        ivr = float(ivr)
    except Exception:
        return "unknown"
    if ivr >= 70:
        return "70+"
    if ivr >= 50:
        return "50-69"
    if ivr >= 30:
        return "30-49"
    return "<30"


def _bucket_rows(rows: list[dict], key_fn) -> dict[str, list[float]]:
    buckets = defaultdict(list)
    for row in rows:
        r = row.get("realized_r")
        if r is None:
            continue
        try:
            buckets[key_fn(row)].append(float(r))
        except Exception:
            buckets["unknown"].append(float(r))
    return buckets


def _print_bucket(title: str, buckets: dict[str, list[float]], min_count: int):
    print(f"\n{title}")
    print("  bucket                n   win%    avgR   totalR   bestR  worstR")
    print("  " + "-" * 62)
    for key, vals in sorted(
        buckets.items(),
        key=lambda item: (len(item[1]), sum(item[1])),
        reverse=True,
    ):
        if len(vals) < min_count:
            continue
        wins = sum(1 for v in vals if v > 0)
        print(
            f"  {str(key)[:18]:<18} "
            f"{len(vals):>4} "
            f"{wins / len(vals) * 100:>6.1f} "
            f"{mean(vals):>7.3f} "
            f"{sum(vals):>8.2f} "
            f"{max(vals):>7.2f} "
            f"{min(vals):>7.2f}"
        )


def _print_count_bucket(title: str, rows: list[dict], key: str):
    counts = defaultdict(int)
    for row in rows:
        counts[row.get(key) or "unknown"] += 1
    if not counts:
        return
    print(f"\n{title}")
    print("  bucket                n")
    print("  " + "-" * 25)
    for bucket, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {str(bucket)[:18]:<18} {count:>4}")


def build_enriched_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        raw = _json_dict(row.get("raw_scanner_data"))
        trade = raw.get("trade") or {}
        row = dict(row)
        row["_entry_hour"] = _entry_hour(row)
        row["_score_band"] = _score_band(row.get("confluence_score"))
        row["_flow_conf"] = _first_flow_conf(raw)
        row["_regime"] = _regime(raw)
        row["_above_vwap"] = _above_vwap(raw)
        row["_ivr_band"] = _ivr_band(row)
        row["_strategy"] = row.get("strategy") or trade.get("strategy") or "unknown"
        row["_direction"] = row.get("direction") or trade.get("direction") or "unknown"
        row["_ticker"] = row.get("ticker") or "unknown"
        enriched.append(row)
    return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Summarize closed trade outcomes for scanner and agent tuning."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum bucket size")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    rows = build_enriched_rows(_load_rows(db_path))
    if not rows:
        print("No closed trades found.")
        return

    realized = [float(r.get("realized_r") or 0) for r in rows]
    winners = sum(1 for r in realized if r > 0)
    print(f"Closed trades: {len(rows)}")
    print(f"Win rate:      {winners / len(rows) * 100:.1f}%")
    print(f"Avg R:         {mean(realized):+.3f}R")
    print(f"Total R:       {sum(realized):+.2f}R")
    print(f"Best/Worst:    {max(realized):+.2f}R / {min(realized):+.2f}R")

    groups = [
        ("By Strategy", lambda r: r["_strategy"]),
        ("By Direction", lambda r: r["_direction"]),
        ("By Ticker", lambda r: r["_ticker"]),
        ("By Entry Hour", lambda r: r["_entry_hour"]),
        ("By Regime", lambda r: r["_regime"]),
        ("By Confluence Band", lambda r: r["_score_band"]),
        ("By IVR Band", lambda r: r["_ivr_band"]),
        ("By Flow Confidence", lambda r: r["_flow_conf"]),
        ("By VWAP At Entry", lambda r: r["_above_vwap"]),
    ]

    for title, key_fn in groups:
        _print_bucket(title, _bucket_rows(rows, key_fn), args.min_count)

    candidates = _load_candidate_rows(db_path)
    if candidates:
        print(f"\nCandidate evaluations logged: {len(candidates)}")
        _print_count_bucket("Candidates By Action", candidates, "action")
        _print_count_bucket("Candidates By Flow Confidence", candidates, "flow_confidence")
        _print_count_bucket("Candidates By Regime", candidates, "regime")


if __name__ == "__main__":
    main()
