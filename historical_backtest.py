from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from features import build_feature_row
from modeling import load_predictor_bundle
from train_models import _build_features


DATASET_PATH = Path("data/player_game_logs.csv")
MARKET_TOLERANCE = {
    "Points": 4.0,
    "Assists": 2.0,
    "Rebounds": 2.0,
    "Points + Rebounds": 4.0,
    "Points + Assists": 4.0,
    "Assists + Rebounds": 3.0,
    "PRA": 5.0,
}


@dataclass(frozen=True)
class HistoricalBacktestResult:
    target_date: str
    min_date: str | None
    max_date: str | None
    available: bool
    game_count: int
    player_count: int
    summary_rows: list[dict[str, Any]]
    player_rows: list[dict[str, Any]]
    best_picks: list[dict[str, Any]] = None
    worst_picks: list[dict[str, Any]] = None


@dataclass(frozen=True)
class BatchBacktestResult:
    start_date: str
    end_date: str
    dates_run: int
    total_predictions: int
    market_summaries: list[dict[str, Any]]
    daily_summaries: list[dict[str, Any]]


@dataclass(frozen=True)
class HistoricalBacktestOverview:
    min_date: str | None
    max_date: str | None
    available_dates: list[str]


def _load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Historical dataset not found at {DATASET_PATH}.")
    frame = pd.read_csv(DATASET_PATH)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame = frame.dropna(subset=["game_date"]).copy()
    return frame


def _combo_actuals(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["Points + Rebounds"] = enriched["points"] + enriched["rebounds"]
    enriched["Points + Assists"] = enriched["points"] + enriched["assists"]
    enriched["Assists + Rebounds"] = enriched["assists"] + enriched["rebounds"]
    enriched["PRA"] = enriched["points"] + enriched["assists"] + enriched["rebounds"]
    return enriched


@lru_cache(maxsize=1)
def get_historical_backtest_overview() -> HistoricalBacktestOverview:
    frame = _load_dataset()
    unique_dates = (
        frame["game_date"]
        .dropna()
        .dt.date
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    available_dates = [value.isoformat() for value in unique_dates[-10:]]
    min_date = unique_dates[0].isoformat() if unique_dates else None
    max_date = unique_dates[-1].isoformat() if unique_dates else None
    return HistoricalBacktestOverview(
        min_date=min_date,
        max_date=max_date,
        available_dates=available_dates,
    )


@lru_cache(maxsize=16)
def run_historical_backtest(target_date: str) -> HistoricalBacktestResult:
    raw_frame = _load_dataset()
    min_date = raw_frame["game_date"].min()
    max_date = raw_frame["game_date"].max()
    ordered_raw = raw_frame.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    feature_frame = _build_features(raw_frame).reset_index(drop=True)
    metadata_columns = [
        column
        for column in ["game_id", "player_name", "team_abbr", "opponent_abbr", "minutes", "home", "rest_days", "is_back_to_back"]
        if column in ordered_raw.columns
    ]
    feature_frame = pd.concat([ordered_raw[metadata_columns], feature_frame], axis=1)
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]
    feature_frame["game_date"] = pd.to_datetime(feature_frame["game_date"], errors="coerce")
    feature_frame = feature_frame.dropna(subset=["game_date"]).copy()

    requested_date = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(requested_date):
        return HistoricalBacktestResult(
            target_date=target_date,
            min_date=min_date.date().isoformat() if pd.notna(min_date) else None,
            max_date=max_date.date().isoformat() if pd.notna(max_date) else None,
            available=False,
            game_count=0,
            player_count=0,
            summary_rows=[],
            player_rows=[],
        )

    day_frame = feature_frame[feature_frame["game_date"].dt.date == requested_date.date()].copy()
    if day_frame.empty:
        return HistoricalBacktestResult(
            target_date=requested_date.date().isoformat(),
            min_date=min_date.date().isoformat() if pd.notna(min_date) else None,
            max_date=max_date.date().isoformat() if pd.notna(max_date) else None,
            available=False,
            game_count=0,
            player_count=0,
            summary_rows=[],
            player_rows=[],
        )

    bundle = load_predictor_bundle()
    predictions_by_target: dict[str, pd.Series] = {}
    for target, spec in bundle.specs.items():
        frame = day_frame.reindex(columns=spec.feature_names or [], fill_value=0.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
            predictions_by_target[target] = pd.Series(bundle.models[target].predict(frame), index=day_frame.index)

    results = day_frame.copy()
    results["pred_points"] = predictions_by_target["points"].astype(float).round(1)
    results["pred_assists"] = predictions_by_target["assists"].astype(float).round(1)
    results["pred_rebounds"] = predictions_by_target["rebounds"].astype(float).round(1)
    results["pred_points_rebounds"] = (results["pred_points"] + results["pred_rebounds"]).round(1)
    results["pred_points_assists"] = (results["pred_points"] + results["pred_assists"]).round(1)
    results["pred_assists_rebounds"] = (results["pred_assists"] + results["pred_rebounds"]).round(1)
    results["pred_pra"] = (results["pred_points"] + results["pred_assists"] + results["pred_rebounds"]).round(1)
    results = _combo_actuals(results)

    metric_map = {
        "Points": ("pred_points", "points"),
        "Assists": ("pred_assists", "assists"),
        "Rebounds": ("pred_rebounds", "rebounds"),
        "Points + Rebounds": ("pred_points_rebounds", "Points + Rebounds"),
        "Points + Assists": ("pred_points_assists", "Points + Assists"),
        "Assists + Rebounds": ("pred_assists_rebounds", "Assists + Rebounds"),
        "PRA": ("pred_pra", "PRA"),
    }

    summary_rows = []
    for market, (pred_col, actual_col) in metric_map.items():
        abs_error = (results[pred_col] - results[actual_col]).abs()
        tolerance = MARKET_TOLERANCE.get(market, 3.0)
        accuracy_pct = ((abs_error <= tolerance).mean() * 100.0) if len(abs_error) else 0.0
        summary_rows.append(
            {
                "market": market,
                "mae": round(float(abs_error.mean()), 2),
                "accuracy_pct": round(float(accuracy_pct), 1),
                "tolerance": tolerance,
                "rows": int(len(results)),
            }
        )

    pts_tol = MARKET_TOLERANCE["Points"]
    ast_tol = MARKET_TOLERANCE["Assists"]
    reb_tol = MARKET_TOLERANCE["Rebounds"]
    pra_tol = MARKET_TOLERANCE["PRA"]

    display_rows = []
    for _, row in results.sort_values(["game_id", "team_abbr", "player_name"]).iterrows():
        pts_err = round(float(row["pred_points"] - row["points"]), 1)
        ast_err = round(float(row["pred_assists"] - row["assists"]), 1)
        reb_err = round(float(row["pred_rebounds"] - row["rebounds"]), 1)
        pra_err = round(float(row["pred_pra"] - row["PRA"]), 1)
        display_rows.append(
            {
                "game_date": row["game_date"].date().isoformat(),
                "game_id": row["game_id"],
                "player_name": row["player_name"],
                "team_abbr": row["team_abbr"],
                "opponent_abbr": row["opponent_abbr"],
                "points_pred": row["pred_points"],
                "points_actual": round(float(row["points"]), 1),
                "points_error": pts_err,
                "points_hit": abs(pts_err) <= pts_tol,
                "assists_pred": row["pred_assists"],
                "assists_actual": round(float(row["assists"]), 1),
                "assists_error": ast_err,
                "assists_hit": abs(ast_err) <= ast_tol,
                "rebounds_pred": row["pred_rebounds"],
                "rebounds_actual": round(float(row["rebounds"]), 1),
                "rebounds_error": reb_err,
                "rebounds_hit": abs(reb_err) <= reb_tol,
                "pra_pred": row["pred_pra"],
                "pra_actual": round(float(row["PRA"]), 1),
                "pra_error": pra_err,
                "pra_hit": abs(pra_err) <= pra_tol,
            }
        )

    # Enrich summary rows with hit counts
    hit_tolerance = {
        "Points": pts_tol, "Assists": ast_tol, "Rebounds": reb_tol,
        "Points + Rebounds": MARKET_TOLERANCE["Points + Rebounds"],
        "Points + Assists": MARKET_TOLERANCE["Points + Assists"],
        "Assists + Rebounds": MARKET_TOLERANCE["Assists + Rebounds"],
        "PRA": pra_tol,
    }
    for sr in summary_rows:
        tol = hit_tolerance.get(sr["market"], 3.0)
        pred_col, actual_col = metric_map[sr["market"]]
        hits = int(((results[pred_col] - results[actual_col]).abs() <= tol).sum())
        sr["hit_count"] = hits
        sr["total"] = len(results)

    # Best picks: smallest absolute error on points
    best = sorted(display_rows, key=lambda r: abs(r["points_error"]))[:5]
    worst = sorted(display_rows, key=lambda r: abs(r["points_error"]), reverse=True)[:5]

    return HistoricalBacktestResult(
        target_date=requested_date.date().isoformat(),
        min_date=min_date.date().isoformat() if pd.notna(min_date) else None,
        max_date=max_date.date().isoformat() if pd.notna(max_date) else None,
        available=True,
        game_count=int(results["game_id"].nunique()),
        player_count=int(results["player_id"].nunique()),
        summary_rows=summary_rows,
        player_rows=display_rows,
        best_picks=best,
        worst_picks=worst,
    )


def run_batch_backtest(start_date: str, end_date: str) -> BatchBacktestResult:
    """Run predictions across all game dates in a range and aggregate accuracy."""
    raw_frame = _load_dataset()
    ordered_raw = raw_frame.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    feature_frame = _build_features(raw_frame).reset_index(drop=True)
    metadata_columns = [
        c for c in ["game_id", "player_name", "team_abbr", "opponent_abbr", "minutes", "home", "rest_days", "is_back_to_back"]
        if c in ordered_raw.columns
    ]
    feature_frame = pd.concat([ordered_raw[metadata_columns], feature_frame], axis=1)
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]
    feature_frame["game_date"] = pd.to_datetime(feature_frame["game_date"], errors="coerce")
    feature_frame = feature_frame.dropna(subset=["game_date"]).copy()

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    range_frame = feature_frame[
        (feature_frame["game_date"].dt.date >= start.date()) &
        (feature_frame["game_date"].dt.date <= end.date())
    ].copy()

    if range_frame.empty:
        return BatchBacktestResult(
            start_date=start_date, end_date=end_date,
            dates_run=0, total_predictions=0,
            market_summaries=[], daily_summaries=[],
        )

    bundle = load_predictor_bundle()
    for target, spec in bundle.specs.items():
        frame = range_frame.reindex(columns=spec.feature_names or [], fill_value=0.0)
        range_frame[f"pred_{target}"] = bundle.models[target].predict(frame)

    range_frame = _combo_actuals(range_frame)
    range_frame["pred_points_rebounds"] = range_frame["pred_points"] + range_frame["pred_rebounds"]
    range_frame["pred_points_assists"] = range_frame["pred_points"] + range_frame["pred_assists"]
    range_frame["pred_assists_rebounds"] = range_frame["pred_assists"] + range_frame["pred_rebounds"]
    range_frame["pred_pra"] = range_frame["pred_points"] + range_frame["pred_assists"] + range_frame["pred_rebounds"]

    metric_map = {
        "Points": ("pred_points", "points"),
        "Assists": ("pred_assists", "assists"),
        "Rebounds": ("pred_rebounds", "rebounds"),
        "Points + Rebounds": ("pred_points_rebounds", "Points + Rebounds"),
        "Points + Assists": ("pred_points_assists", "Points + Assists"),
        "Assists + Rebounds": ("pred_assists_rebounds", "Assists + Rebounds"),
        "PRA": ("pred_pra", "PRA"),
    }

    market_summaries = []
    for market, (pred_col, actual_col) in metric_map.items():
        tol = MARKET_TOLERANCE.get(market, 3.0)
        err = (range_frame[pred_col] - range_frame[actual_col]).abs()
        hit_count = int((err <= tol).sum())
        total = len(err)
        market_summaries.append({
            "market": market,
            "mae": round(float(err.mean()), 2),
            "hit_count": hit_count,
            "total": total,
            "hit_rate": round(hit_count / total * 100, 1) if total else 0.0,
            "tolerance": tol,
        })

    daily_summaries = []
    for date, day_df in range_frame.groupby(range_frame["game_date"].dt.date):
        pts_err = (day_df["pred_points"] - day_df["points"]).abs()
        ast_err = (day_df["pred_assists"] - day_df["assists"]).abs()
        reb_err = (day_df["pred_rebounds"] - day_df["rebounds"]).abs()
        daily_summaries.append({
            "date": date.isoformat(),
            "players": len(day_df),
            "pts_hit_rate": round((pts_err <= MARKET_TOLERANCE["Points"]).mean() * 100, 1),
            "ast_hit_rate": round((ast_err <= MARKET_TOLERANCE["Assists"]).mean() * 100, 1),
            "reb_hit_rate": round((reb_err <= MARKET_TOLERANCE["Rebounds"]).mean() * 100, 1),
            "pts_mae": round(float(pts_err.mean()), 2),
            "ast_mae": round(float(ast_err.mean()), 2),
            "reb_mae": round(float(reb_err.mean()), 2),
        })

    unique_dates = range_frame["game_date"].dt.date.nunique()
    return BatchBacktestResult(
        start_date=start_date,
        end_date=end_date,
        dates_run=int(unique_dates),
        total_predictions=len(range_frame),
        market_summaries=market_summaries,
        daily_summaries=sorted(daily_summaries, key=lambda r: r["date"]),
    )


_UNDERDOG_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 40.0}

_DISCREPANCY_MARKETS = [
    ("Points",   "pred_points",   "points",   "points_rolling_10",   2.0),
    ("Assists",  "pred_assists",  "assists",  "assists_rolling_10",  0.75),
    ("Rebounds", "pred_rebounds", "rebounds", "rebounds_rolling_10", 0.75),
]


def run_discrepancy_parlay_sim(
    start_date: str,
    end_date: str,
    min_edge: float | None = None,
) -> dict[str, Any]:
    """
    Identify discrepancy plays where model projects significantly above the
    10-game rolling average (used as Underdog line proxy), then simulate
    all possible 2/3/4/5/6-pick same-day parlays built from those picks.

    A pick hits when actual > rolling-10 line.
    A parlay wins when every pick hits.
    """
    raw_frame = _load_dataset()
    ordered_raw = raw_frame.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    feature_frame = _build_features(raw_frame).reset_index(drop=True)
    metadata_columns = [
        c for c in ["game_id", "player_name", "team_abbr", "opponent_abbr", "minutes"]
        if c in ordered_raw.columns
    ]
    feature_frame = pd.concat([ordered_raw[metadata_columns], feature_frame], axis=1)
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]
    feature_frame["game_date"] = pd.to_datetime(feature_frame["game_date"], errors="coerce")
    feature_frame = feature_frame.dropna(subset=["game_date"]).copy()

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    range_frame = feature_frame[
        (feature_frame["game_date"].dt.date >= start.date()) &
        (feature_frame["game_date"].dt.date <= end.date())
    ].copy()

    if range_frame.empty:
        return {"error": "No data in range.", "picks": [], "parlay_results": []}

    bundle = load_predictor_bundle()
    for target, spec in bundle.specs.items():
        frame = range_frame.reindex(columns=spec.feature_names or [], fill_value=0.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
            range_frame[f"pred_{target}"] = bundle.models[target].predict(frame)

    # Build individual discrepancy picks
    picks: list[dict[str, Any]] = []
    for market, pred_col, actual_col, line_col, default_min_edge in _DISCREPANCY_MARKETS:
        threshold = min_edge if min_edge is not None else default_min_edge
        if line_col not in range_frame.columns or pred_col not in range_frame.columns:
            continue
        sub = range_frame[[
            "game_date", "player_name", "team_abbr", "opponent_abbr",
            pred_col, actual_col, line_col,
        ]].copy()
        sub = sub[sub[line_col] > 0].copy()
        sub["edge"] = sub[pred_col] - sub[line_col]
        sub = sub[sub["edge"] >= threshold].copy()
        sub["hit"] = sub[actual_col] > sub[line_col]
        for _, row in sub.iterrows():
            picks.append({
                "date": row["game_date"].date().isoformat(),
                "player": row["player_name"],
                "team": row["team_abbr"],
                "opp": row["opponent_abbr"],
                "market": market,
                "line": round(float(row[line_col]), 1),
                "model_proj": round(float(row[pred_col]), 1),
                "edge": round(float(row["edge"]), 1),
                "actual": round(float(row[actual_col]), 1),
                "hit": bool(row["hit"]),
            })

    if not picks:
        return {"error": "No discrepancy picks found.", "picks": [], "parlay_results": []}

    total_picks = len(picks)
    total_hits = sum(1 for p in picks if p["hit"])

    # Group by date for same-day parlay construction
    by_date: dict[str, list[dict]] = {}
    for pick in picks:
        by_date.setdefault(pick["date"], []).append(pick)

    parlay_results: list[dict[str, Any]] = []
    for size, payout in _UNDERDOG_PAYOUTS.items():
        total_parlays = 0
        winning_parlays = 0
        for date_picks in by_date.values():
            if len(date_picks) < size:
                continue
            for combo in itertools.combinations(date_picks, size):
                total_parlays += 1
                if all(p["hit"] for p in combo):
                    winning_parlays += 1

        if total_parlays == 0:
            continue

        win_rate = winning_parlays / total_parlays
        ev_per_dollar = win_rate * payout - 1.0
        roi_pct = ev_per_dollar * 100.0
        parlay_results.append({
            "size": size,
            "payout": payout,
            "total_parlays": total_parlays,
            "winning_parlays": winning_parlays,
            "win_rate_pct": round(win_rate * 100, 1),
            "ev_per_dollar": round(ev_per_dollar, 3),
            "roi_pct": round(roi_pct, 1),
            "break_even_pct": round((1.0 / payout) * 100, 1),
        })

    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_picks": total_picks,
        "pick_hit_rate_pct": round(total_hits / total_picks * 100, 1),
        "picks": sorted(picks, key=lambda p: (p["date"], -p["edge"])),
        "parlay_results": parlay_results,
    }
