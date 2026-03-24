"""
Standalone script — safe to run via cron.
Fetches Underdog + PrizePicks lines, computes discrepancies, appends to
data/discrepancy_history.csv (one snapshot per calendar day).

Cron examples (add via `crontab -e`):
  # Scrape lines at 11am, 2pm, and 6pm daily
  0 11,14,18 * * * /path/to/.venv/bin/python /path/to/snapshot_discrepancies.py

  # Ingest latest game results nightly at 2am
  0 2 * * * /path/to/.venv/bin/python /path/to/data_ingest.py --seasons 2025-26

  # Retrain models every Monday at 3am
  0 3 * * 1 /path/to/.venv/bin/python /path/to/train_models.py
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock

import prizepicks_client
import underdog_client

HISTORY_PATH = Path("data/discrepancy_history.csv")
FIELDS = ["snapshot_date", "player_name", "market", "opponent_abbr", "ud_line", "pp_line", "diff", "bet"]
MIN_DIFF = 0.5
_lock = Lock()


def _norm_name(v: str) -> str:
    return " ".join(v.lower().replace(".", "").split())


def _norm_market(v: str) -> str:
    return v.strip().lower()


def fetch_discrepancies() -> list[dict]:
    pp_entries = prizepicks_client.PrizePicksClient().fetch_board_entries()
    ud_entries = underdog_client.UnderdogClient().fetch_board_entries()

    pp_lookup = {(_norm_name(e.player_name), _norm_market(e.market_label)): e for e in pp_entries}

    discrepancies = []
    for ud in ud_entries:
        key = (_norm_name(ud.player_name), _norm_market(ud.market_label))
        pp = pp_lookup.get(key)
        if pp is None:
            continue
        abs_diff = abs(ud.line_score - pp.line_score)
        if abs_diff < MIN_DIFF:
            continue
        bet = "OVER" if ud.line_score < pp.line_score else "UNDER"
        discrepancies.append({
            "player_name": ud.player_name,
            "market": ud.market_label,
            "opponent_abbr": ud.opponent_abbr or pp.opponent_abbr or "N/A",
            "ud_line": round(ud.line_score, 1),
            "pp_line": round(pp.line_score, 1),
            "diff": round(abs_diff, 1),
            "bet": bet,
        })

    discrepancies.sort(key=lambda r: -r["diff"])
    return discrepancies


def save_snapshot(discrepancies: list[dict]) -> bool:
    """Append today's snapshot. Returns True if saved, False if already exists."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _lock:
        existing_dates: set[str] = set()
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    existing_dates.add(row.get("snapshot_date", ""))
        if today in existing_dates:
            return False
        write_header = not HISTORY_PATH.exists()
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            for d in discrepancies:
                writer.writerow({"snapshot_date": today, **{k: d[k] for k in FIELDS[1:]}})
    return True


if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Fetching lines...")
    try:
        picks = fetch_discrepancies()
    except Exception as exc:
        print(f"ERROR fetching lines: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(picks)} discrepancies")
    saved = save_snapshot(picks)
    if saved:
        print(f"  Saved snapshot to {HISTORY_PATH}")
    else:
        print(f"  Snapshot for today already exists — skipped")

    # Print top 10 by gap
    print()
    print(f"  {'PLAYER':<28} {'MKT':<10} {'OPP':<5} {'BET':<6} {'UD':>6} {'PP':>6} {'GAP':>4}")
    print("  " + "-" * 66)
    for r in picks[:10]:
        print(f"  {r['player_name']:<28} {r['market']:<10} {r['opponent_abbr']:<5} {r['bet']:<6} {r['ud_line']:>6.1f} {r['pp_line']:>6.1f} {r['diff']:>4.1f}")
