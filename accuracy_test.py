"""
Standalone accuracy test for points, assists, and rebounds models.
Runs on the last 20% of the dataset (validation period) and prints a results table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from modeling import load_predictor_bundle
from train_models import _build_features, TARGET_COLUMNS, AUXILIARY_TARGETS


DATASET_PATH = Path("data/player_game_logs.csv")
VALIDATION_RATIO = 0.20
TARGETS = ["points", "assists", "rebounds"]
TOLERANCES = {
    "points": [2, 3, 5],
    "assists": [1, 2, 3],
    "rebounds": [1, 2, 3],
}


def _bar(value: float, width: int = 20, char: str = "█") -> str:
    filled = int(round(value * width))
    return char * filled + "░" * (width - filled)


def run_accuracy_test() -> None:
    print("\n" + "=" * 62)
    print("  NBA PREDICTOR — ACCURACY TEST")
    print("=" * 62)

    print("\nLoading dataset...")
    frame = pd.read_csv(DATASET_PATH)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame = frame.dropna(subset=["game_date"]).copy()

    print("Building features...")
    feature_frame = _build_features(frame)
    feature_frame = feature_frame.dropna(subset=TARGETS).copy()
    feature_frame = feature_frame.sort_values("game_date").reset_index(drop=True)

    split_idx = int(len(feature_frame) * (1 - VALIDATION_RATIO))
    val = feature_frame.iloc[split_idx:].copy()

    date_min = val["game_date"].min().date()
    date_max = val["game_date"].max().date()
    print(f"Validation set: {len(val):,} rows | {date_min} → {date_max}")

    print("Loading models...")
    bundle = load_predictor_bundle()

    # Prop-relevant filter: only games where player played meaningful minutes
    val_full = val.copy()
    val_prop = val[val["minutes"] >= 15].copy()
    print(f"  All rows:        {len(val_full):,}")
    print(f"  Prop-relevant (≥15 min): {len(val_prop):,} ({len(val_prop)/len(val_full)*100:.1f}% of set)")

    results: dict[str, dict] = {}
    for target in TARGETS:
        spec = bundle.specs[target]

        def _eval(subset: pd.DataFrame) -> dict:
            X = subset.reindex(columns=spec.feature_names or [], fill_value=0.0)
            y = subset[target]
            preds = pd.Series(bundle.models[target].predict(X), index=subset.index)
            abs_err = (y - preds).abs()
            within = {tol: float((abs_err <= tol).mean()) for tol in TOLERANCES[target]}
            median_line = y.median()
            return {
                "mae":     mean_absolute_error(y, preds),
                "rmse":    root_mean_squared_error(y, preds),
                "r2":      r2_score(y, preds),
                "within":  within,
                "dir_acc": float(((preds > median_line) == (y > median_line)).mean()),
                "bias":    float((preds - y).mean()),
                "n":       len(y),
            }

        results[target] = {
            "all":  _eval(val_full),
            "prop": _eval(val_prop),
        }

    # ── Print results ──────────────────────────────────────────
    baseline = {"points": 4.40, "assists": 1.33, "rebounds": 1.88}
    print()
    for target in TARGETS:
        ra = results[target]["all"]
        rp = results[target]["prop"]
        label = target.upper()
        print(f"┌─ {label} {'─' * (55 - len(label))}┐")
        print(f"│  ALL GAMES (n={ra['n']:,}):                               │")
        print(f"│    MAE {ra['mae']:>6.3f}  RMSE {ra['rmse']:>6.3f}  R² {ra['r2']:>5.3f}  Bias {ra['bias']:>+5.2f}  │")
        print(f"│  PROP-RELEVANT ≥15 min (n={rp['n']:,}):                   │")
        print(f"│    MAE {rp['mae']:>6.3f}  RMSE {rp['rmse']:>6.3f}  R² {rp['r2']:>5.3f}  Bias {rp['bias']:>+5.2f}  │")
        print(f"│  Dir. accuracy: {rp['dir_acc']*100:>5.1f}%   Within-tolerance:        │")
        for tol, rate in rp["within"].items():
            bar = _bar(rate, width=15)
            print(f"│    ±{tol}  {bar}  {rate*100:>5.1f}%                    │")
        print(f"└{'─' * 58}┘")
        print()

    # ── Summary table ──────────────────────────────────────────
    print("─" * 68)
    print(f"  {'TARGET':<12} {'ALL MAE':>9}  {'PROP MAE':>9}  {'Δ vs BASE':>10}  {'DIR ACC':>9}")
    print("─" * 68)
    for target in TARGETS:
        ra = results[target]["all"]
        rp = results[target]["prop"]
        delta = rp["mae"] - baseline[target]
        delta_str = f"{delta:+.3f}"
        print(
            f"  {target.capitalize():<12}"
            f" {ra['mae']:>9.3f}"
            f"  {rp['mae']:>9.3f}"
            f"  {delta_str:>10}"
            f"  {rp['dir_acc']*100:>8.1f}%"
        )
    print("─" * 68)
    print("  Prop MAE = accuracy on games ≥15 min (actual prop-eligible games)")
    print()

    # ── Underdog Fantasy ROI breakdown ────────────────────────
    print("=" * 62)
    print("  UNDERDOG FANTASY — ROI ANALYSIS")
    print("=" * 62)

    # Underdog Pick'em payouts (standard balanced lines)
    underdog_entries = {
        2: {"payout": 3.0,  "label": "2-pick"},
        3: {"payout": 6.0,  "label": "3-pick"},
        4: {"payout": 10.0, "label": "4-pick"},
        5: {"payout": 20.0, "label": "5-pick"},
    }

    print("\n  Break-even accuracy needed per pick vs your model:")
    print(f"  {'ENTRY':<10} {'PAYOUT':>8}  {'BREAK-EVEN':>12}  {'PTS EDGE':>10}  {'AST EDGE':>10}  {'REB EDGE':>10}")
    print("  " + "─" * 62)
    for n, cfg in underdog_entries.items():
        payout = cfg["payout"]
        break_even = (1.0 / payout) ** (1.0 / n)  # per-pick accuracy needed
        for target in TARGETS:
            rp = results[target]["prop"]
            rp[f"edge_{n}"] = rp["dir_acc"] - break_even

        pts_edge = results["points"]["prop"][f"edge_{n}"] * 100
        ast_edge = results["assists"]["prop"][f"edge_{n}"] * 100
        reb_edge = results["rebounds"]["prop"][f"edge_{n}"] * 100

        def _edge_str(e: float) -> str:
            return f"+{e:.1f}%" if e >= 0 else f"{e:.1f}%"

        print(
            f"  {cfg['label']:<10} {payout:>6.0f}x"
            f"  {break_even*100:>10.1f}%"
            f"  {_edge_str(pts_edge):>10}"
            f"  {_edge_str(ast_edge):>10}"
            f"  {_edge_str(reb_edge):>10}"
        )
    print()

    # Per-target expected ROI on a flat 2-pick entry
    print("  Expected ROI on $10 flat entries (2-pick, 3x payout):")
    print(f"  {'TARGET':<12} {'DIR ACC':>9}  {'WIN RATE (2p)':>14}  {'EV per $10':>12}  {'ROI':>8}")
    print("  " + "─" * 54)
    for target in TARGETS:
        rp = results[target]["prop"]
        d = rp["dir_acc"]
        win_rate_2pick = d ** 2           # both picks hit (assuming independence)
        ev = win_rate_2pick * 30.0 - 10.0 # $10 entry, $30 payout
        roi = ev / 10.0 * 100
        print(
            f"  {target.capitalize():<12}"
            f" {d*100:>8.1f}%"
            f"  {win_rate_2pick*100:>12.1f}%"
            f"  ${ev:>10.2f}"
            f"  {roi:>7.1f}%"
        )

    print()
    print("  Best market for Underdog single-target 2-picks:")
    best = max(TARGETS, key=lambda t: results[t]["prop"]["dir_acc"])
    rp = results[best]["prop"]
    d = rp["dir_acc"]
    print(f"  → {best.upper()} at {d*100:.1f}% directional accuracy")
    print(f"    Both picks hit {d**2*100:.1f}% of the time → EV ${d**2*30-10:.2f} per $10 entry")
    print()


if __name__ == "__main__":
    run_accuracy_test()
