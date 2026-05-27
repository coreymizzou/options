#!/usr/bin/env python3
"""
Label candidate_evaluations from timestamped candidate_price_snapshots.

This script deliberately avoids fetching a "current" quote to backfill old
outcomes. Labels are only written when a stored snapshot exists near the target
time, so 1h/EOD/1d labels mean what they say.
"""

import argparse
import sqlite3
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_dt(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _nearest_snapshot(
    snapshots: List[sqlite3.Row],
    target: datetime,
    tolerance_minutes: int,
) -> Optional[sqlite3.Row]:
    best: Optional[Tuple[float, sqlite3.Row]] = None
    for snap in snapshots:
        ts = _parse_dt(snap["timestamp"])
        if not ts:
            continue
        diff = abs((ts - target).total_seconds())
        if diff <= tolerance_minutes * 60:
            if best is None or diff < best[0]:
                best = (diff, snap)
    return best[1] if best else None


def _r(entry_price: float, option_price: float) -> float:
    return round((float(option_price) - entry_price) / entry_price, 4)


def _snapshot_rows(con: sqlite3.Connection, candidate_id: int) -> List[sqlite3.Row]:
    return con.execute(
        """
        SELECT *
        FROM candidate_price_snapshots
        WHERE candidate_id = ?
        ORDER BY timestamp
        """,
        (candidate_id,),
    ).fetchall()


def _eod_target(ts: datetime) -> datetime:
    return datetime.combine(ts.date(), dtime(hour=15, minute=45))


def label_rows(
    db_path: Path,
    limit: int,
    tolerance_1h_minutes: int,
    tolerance_eod_minutes: int,
    tolerance_1d_minutes: int,
) -> int:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        """
        SELECT *
        FROM candidate_evaluations
        WHERE entry_price IS NOT NULL
          AND (
            outcome_1h_r IS NULL OR outcome_eod_r IS NULL OR outcome_1d_r IS NULL
            OR mfe_r IS NULL OR mae_r IS NULL
          )
        ORDER BY timestamp
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    now = datetime.now()
    labeled = 0
    skipped_no_snapshots = 0
    skipped_not_ready = 0

    for row in rows:
        candidate_ts = _parse_dt(row["timestamp"])
        entry_price = float(row["entry_price"] or 0)
        if not candidate_ts or entry_price <= 0:
            continue

        snapshots = _snapshot_rows(con, row["id"])
        if not snapshots:
            skipped_no_snapshots += 1
            continue

        updates: Dict[str, object] = {}
        observed_rs = [
            _r(entry_price, snap["option_price"])
            for snap in snapshots
            if snap["option_price"] is not None
        ]
        if observed_rs:
            updates["mfe_r"] = max(observed_rs)
            updates["mae_r"] = min(observed_rs)

        target_1h = candidate_ts + timedelta(hours=1)
        if row["outcome_1h_r"] is None:
            if now >= target_1h:
                snap = _nearest_snapshot(snapshots, target_1h, tolerance_1h_minutes)
                if snap:
                    updates["outcome_1h_r"] = _r(entry_price, snap["option_price"])
            else:
                skipped_not_ready += 1

        target_eod = _eod_target(candidate_ts)
        if row["outcome_eod_r"] is None:
            if now >= target_eod:
                snap = _nearest_snapshot(snapshots, target_eod, tolerance_eod_minutes)
                if snap:
                    updates["outcome_eod_r"] = _r(entry_price, snap["option_price"])
            else:
                skipped_not_ready += 1

        target_1d = candidate_ts + timedelta(hours=24)
        if row["outcome_1d_r"] is None:
            if now >= target_1d:
                snap = _nearest_snapshot(snapshots, target_1d, tolerance_1d_minutes)
                if snap:
                    updates["outcome_1d_r"] = _r(entry_price, snap["option_price"])
            else:
                skipped_not_ready += 1

        if updates:
            updates["outcome_labeled_at"] = now.isoformat()
            sets = ", ".join(f"{key} = ?" for key in updates)
            con.execute(
                f"UPDATE candidate_evaluations SET {sets} WHERE id = ?",
                [*updates.values(), row["id"]],
            )
            labeled += 1

    con.commit()
    con.close()
    print(f"Labeled/updated {labeled} candidate rows from {db_path}")
    print(f"Skipped without snapshots: {skipped_no_snapshots}")
    print(f"Skipped not ready yet: {skipped_not_ready}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Label candidate outcomes from stored price snapshots"
    )
    parser.add_argument("--db", default="scanner_data_live.db", help="SQLite DB path")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to process")
    parser.add_argument("--tolerance-1h-minutes", type=int, default=15)
    parser.add_argument("--tolerance-eod-minutes", type=int, default=20)
    parser.add_argument("--tolerance-1d-minutes", type=int, default=90)
    args = parser.parse_args()
    return label_rows(
        Path(args.db).resolve(),
        args.limit,
        args.tolerance_1h_minutes,
        args.tolerance_eod_minutes,
        args.tolerance_1d_minutes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
