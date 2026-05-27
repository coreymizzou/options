#!/usr/bin/env python3
"""
Label candidate_evaluations with forward option-price outcomes.

Run this during/after the session. Repeated runs improve MFE/MAE tracking and
fill 1h/EOD/1d outcome columns once candidates are old enough.
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, Optional

import requests


def _json(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _get_state(con, key: str) -> str:
    row = con.execute("SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
    if not row:
        return ""
    try:
        return json.loads(row[0])
    except Exception:
        return str(row[0] or "")


def _fetch_mid(api_key: str, ticker: str, expiration: str, strike: float, option_type: str) -> Optional[float]:
    try:
        r = requests.get(
            "https://api.tradier.com/v1/markets/options/chains",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            params={"symbol": ticker, "expiration": expiration, "greeks": "false"},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        options = r.json().get("options", {}).get("option", []) or []
        for contract in options:
            if (
                abs(float(contract.get("strike", 0) or 0) - float(strike)) < 0.01
                and str(contract.get("option_type", "")).lower() == option_type.lower()
            ):
                bid = float(contract.get("bid", 0) or 0)
                ask = float(contract.get("ask", 0) or 0)
                last = float(contract.get("last", 0) or 0)
                if bid > 0 and ask > 0:
                    return round((bid + ask) / 2, 2)
                if last > 0:
                    return round(last, 2)
    except Exception:
        return None
    return None


def _candidate_current_price(api_key: str, row: sqlite3.Row) -> Optional[float]:
    raw = _json(row["raw_scanner_data"])
    trade = raw.get("trade", {}) or {}
    main = trade.get("main_leg", {}) or {}
    short = trade.get("short_leg", {}) or {}

    ticker = row["ticker"] or raw.get("ticker")
    expiration = row["expiration"] or main.get("exp") or trade.get("exp")
    strategy = str(row["strategy"] or trade.get("strategy") or "").upper()
    strike = row["strike"] or main.get("strike")
    option_type = main.get("option_type")
    if not option_type:
        option_type = "put" if ("BEAR" in strategy or "PUT" in strategy) else "call"

    if not ticker or not expiration or not strike:
        return None

    long_mid = _fetch_mid(api_key, ticker, expiration, float(strike), option_type)
    if long_mid is None:
        return None

    if "SPREAD" in strategy and short:
        short_strike = short.get("strike")
        short_type = short.get("option_type") or option_type
        if short_strike:
            short_mid = _fetch_mid(api_key, ticker, expiration, float(short_strike), short_type)
            if short_mid is not None:
                return max(round(long_mid - short_mid, 2), 0.01)

    return long_mid


def _is_eod_candidate(ts: datetime, now: datetime) -> bool:
    return ts.date() == now.date() and now.time() >= dtime(hour=15, minute=45)


def label_rows(db_path: Path, limit: int) -> int:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    api_key = os.environ.get("TRADIER_API_KEY") or _get_state(con, "tradier_api_key")
    if not api_key:
        print("No Tradier API key found in env or DB system_state.")
        return 1

    rows = con.execute(
        """
        SELECT *
        FROM candidate_evaluations
        WHERE raw_scanner_data IS NOT NULL
          AND entry_price IS NOT NULL
          AND expiration IS NOT NULL
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
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except Exception:
            continue
        age_hours = (now - ts).total_seconds() / 3600
        current_price = _candidate_current_price(api_key, row)
        entry_price = float(row["entry_price"] or 0)
        if current_price is None or entry_price <= 0:
            continue

        current_r = round((current_price - entry_price) / entry_price, 4)
        updates: Dict[str, float] = {}
        if row["outcome_1h_r"] is None and age_hours >= 1:
            updates["outcome_1h_r"] = current_r
        if row["outcome_eod_r"] is None and _is_eod_candidate(ts, now):
            updates["outcome_eod_r"] = current_r
        if row["outcome_1d_r"] is None and age_hours >= 20:
            updates["outcome_1d_r"] = current_r

        mfe = row["mfe_r"]
        mae = row["mae_r"]
        updates["mfe_r"] = current_r if mfe is None else max(float(mfe), current_r)
        updates["mae_r"] = current_r if mae is None else min(float(mae), current_r)
        updates["outcome_labeled_at"] = now.isoformat()

        if updates:
            sets = ", ".join(f"{k} = ?" for k in updates)
            con.execute(
                f"UPDATE candidate_evaluations SET {sets} WHERE id = ?",
                [*updates.values(), row["id"]],
            )
            labeled += 1

    con.commit()
    con.close()
    print(f"Labeled/updated {labeled} candidate rows from {db_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Label candidate outcomes using current option quotes")
    parser.add_argument("--db", default="scanner_data_live.db", help="SQLite DB path")
    parser.add_argument("--limit", type=int, default=250, help="Max rows to process")
    args = parser.parse_args()
    return label_rows(Path(args.db).resolve(), args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
