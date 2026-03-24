from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline


TARGET_COLUMNS = ["points", "assists", "rebounds"]
AUXILIARY_TARGETS = ["minutes"]
REQUIRED_COLUMNS = ["game_date", "player_id", "points", "assists", "rebounds", "minutes", "home", "rest_days", "is_back_to_back"]
OPTIONAL_NUMERIC_DEFAULTS = {
    "starter": 0.0,
    "fg_pct": 0.0,
    "three_pt_pct": 0.0,
    "ft_pct": 0.0,
    "usage_rate": 0.0,
    "true_shooting_pct": 0.0,
    "player_pace": 0.0,
    "player_off_rating": 0.0,
    "player_def_rating": 0.0,
    "steals": 0.0,
    "blocks": 0.0,
    "turnovers": 0.0,
    "off_rebounds": 0.0,
    "def_rebounds": 0.0,
    "plus_minus": 0.0,
    "team_pace": 0.0,
    "team_off_rating": 0.0,
    "team_def_rating": 0.0,
    "opp_pace": 0.0,
    "opp_off_rating": 0.0,
    "opp_def_rating": 0.0,
    "line_points": 0.0,
    "line_assists": 0.0,
    "line_rebounds": 0.0,
    "closing_over_price": 0.0,
    "closing_under_price": 0.0,
    "team_points_form_5": 0.0,
    "team_assists_form_5": 0.0,
    "team_rebounds_form_5": 0.0,
    "team_points_allowed_5": 0.0,
    "team_assists_allowed_5": 0.0,
    "team_rebounds_allowed_5": 0.0,
    "team_win_pct_10": 0.0,
    "opp_points_form_5": 0.0,
    "opp_assists_form_5": 0.0,
    "opp_rebounds_form_5": 0.0,
    "opp_points_allowed_5": 0.0,
    "opp_assists_allowed_5": 0.0,
    "opp_rebounds_allowed_5": 0.0,
    "opp_win_pct_10": 0.0,
}
PREGAME_CONTEXT_COLUMNS = [
    "home",
    "rest_days",
    "is_back_to_back",
    "team_pace",
    "team_off_rating",
    "team_def_rating",
    "opp_pace",
    "opp_off_rating",
    "opp_def_rating",
    "line_points",
    "line_assists",
    "line_rebounds",
    "closing_over_price",
    "closing_under_price",
    "team_points_form_5",
    "team_assists_form_5",
    "team_rebounds_form_5",
    "team_points_allowed_5",
    "team_assists_allowed_5",
    "team_rebounds_allowed_5",
    "team_win_pct_10",
    "opp_points_form_5",
    "opp_assists_form_5",
    "opp_rebounds_form_5",
    "opp_points_allowed_5",
    "opp_assists_allowed_5",
    "opp_rebounds_allowed_5",
    "opp_win_pct_10",
]
TARGET_CONFIG = {
    "points": {
        "line_column": "line_points",
        "tolerances": [2, 3, 5],
        "candidate_params": [
            {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 30},
            {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 25},
            {"n_estimators": 700, "learning_rate": 0.015, "num_leaves": 63, "min_child_samples": 30, "reg_alpha": 0.1, "reg_lambda": 0.1},
            {"n_estimators": 900, "learning_rate": 0.01, "num_leaves": 47, "min_child_samples": 40, "reg_alpha": 0.05, "reg_lambda": 0.2},
        ],
    },
    "assists": {
        "line_column": "line_assists",
        "tolerances": [1, 2, 3],
        "candidate_params": [
            {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 20},
            {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20},
            {"n_estimators": 700, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 35},
            {"n_estimators": 900, "learning_rate": 0.015, "num_leaves": 47, "min_child_samples": 40},
            {"n_estimators": 600, "learning_rate": 0.02, "num_leaves": 47, "min_child_samples": 30, "reg_alpha": 0.1, "reg_lambda": 0.1},
        ],
    },
    "rebounds": {
        "line_column": "line_rebounds",
        "tolerances": [1, 2, 3],
        "candidate_params": [
            {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 20},
            {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20},
            {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 25, "reg_alpha": 0.05, "reg_lambda": 0.1},
            {"n_estimators": 700, "learning_rate": 0.015, "num_leaves": 47, "min_child_samples": 35, "reg_alpha": 0.1, "reg_lambda": 0.1},
        ],
    },
}
AUXILIARY_TARGET_CONFIG = {
    "minutes": {
        "tolerances": [2, 4, 6],
        "candidate_params": [
            {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 20},
            {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 47, "min_child_samples": 35},
        ],
    }
}


def _add_team_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    team_games = (
        frame.groupby(["game_id", "game_date", "team_abbr", "opponent_abbr"], as_index=False)
        .agg(
            team_points=("points", "sum"),
            team_assists=("assists", "sum"),
            team_rebounds=("rebounds", "sum"),
            team_win=("win", "max"),
        )
        .sort_values(["team_abbr", "game_date", "game_id"])
    )

    opponent_totals = team_games[["game_id", "team_abbr", "team_points", "team_assists", "team_rebounds"]].rename(
        columns={
            "team_abbr": "opponent_abbr",
            "team_points": "team_points_allowed",
            "team_assists": "team_assists_allowed",
            "team_rebounds": "team_rebounds_allowed",
        }
    )
    team_games = team_games.merge(opponent_totals, on=["game_id", "opponent_abbr"], how="left")

    feature_windows = {
        "team_points_form_5": ("team_points", 5),
        "team_assists_form_5": ("team_assists", 5),
        "team_rebounds_form_5": ("team_rebounds", 5),
        "team_points_allowed_5": ("team_points_allowed", 5),
        "team_assists_allowed_5": ("team_assists_allowed", 5),
        "team_rebounds_allowed_5": ("team_rebounds_allowed", 5),
        "team_win_pct_10": ("team_win", 10),
    }
    for feature_name, (source_column, window) in feature_windows.items():
        team_games[feature_name] = (
            team_games.groupby("team_abbr")[source_column]
            .transform(lambda series: series.shift(1).rolling(window, min_periods=1).mean())
        )

    own_team_context = team_games[
        [
            "game_id",
            "team_abbr",
            "team_points_form_5",
            "team_assists_form_5",
            "team_rebounds_form_5",
            "team_points_allowed_5",
            "team_assists_allowed_5",
            "team_rebounds_allowed_5",
            "team_win_pct_10",
        ]
    ]
    opponent_context = own_team_context.rename(
        columns={
            "team_abbr": "opponent_abbr",
            "team_points_form_5": "opp_points_form_5",
            "team_assists_form_5": "opp_assists_form_5",
            "team_rebounds_form_5": "opp_rebounds_form_5",
            "team_points_allowed_5": "opp_points_allowed_5",
            "team_assists_allowed_5": "opp_assists_allowed_5",
            "team_rebounds_allowed_5": "opp_rebounds_allowed_5",
            "team_win_pct_10": "opp_win_pct_10",
        }
    )

    enriched = frame.merge(own_team_context, on=["game_id", "team_abbr"], how="left")
    enriched = enriched.merge(opponent_context, on=["game_id", "opponent_abbr"], how="left")
    return enriched


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["player_id", "game_date"]).copy()
    ordered = _add_team_context_features(ordered)
    for column_name, default_value in OPTIONAL_NUMERIC_DEFAULTS.items():
        if column_name not in ordered.columns:
            ordered[column_name] = default_value
        ordered[column_name] = pd.to_numeric(ordered[column_name], errors="coerce").fillna(default_value)

    engineered_columns = {}
    stat_sources = {
        "points": "points",
        "assists": "assists",
        "rebounds": "rebounds",
        "minutes": "minutes",
        "fg_pct": "fg_pct",
        "three_pt_pct": "three_pt_pct",
        "ft_pct": "ft_pct",
        "usage_rate": "usage_rate",
        "true_shooting_pct": "true_shooting_pct",
        "player_pace": "player_pace",
        "opp_def_rating": "opp_def_rating",
        "opp_pace": "opp_pace",
        "steals": "steals",
        "blocks": "blocks",
        "turnovers": "turnovers",
        "off_rebounds": "off_rebounds",
        "def_rebounds": "def_rebounds",
        "plus_minus": "plus_minus",
        "player_off_rating": "player_off_rating",
        "player_def_rating": "player_def_rating",
        "starter": "starter",
    }
    role_signals = {
        "lead_role_game": (ordered["minutes"] >= 30).astype(float),
        "heavy_load_game": (ordered["minutes"] >= 34).astype(float),
        "rotation_role_game": (ordered["minutes"] >= 24).astype(float),
        "bench_risk_game": (ordered["minutes"] < 20).astype(float),
    }
    for column_name, values in role_signals.items():
        ordered[column_name] = values

    for feature_name, source_column in stat_sources.items():
        for window in (3, 5, 10):
            engineered_columns[f"{feature_name}_rolling_{window}"] = (
                ordered.groupby("player_id")[source_column]
                .transform(lambda series: series.shift(1).rolling(window, min_periods=1).mean())
            )
            engineered_columns[f"{feature_name}_rolling_std_{window}"] = (
                ordered.groupby("player_id")[source_column]
                .transform(lambda series: series.shift(1).rolling(window, min_periods=2).std())
            )
        engineered_columns[f"{feature_name}_season_avg"] = (
            ordered.groupby("player_id")[source_column]
            .transform(lambda series: series.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )
        engineered_columns[f"{feature_name}_last_game"] = ordered.groupby("player_id")[source_column].shift(1)

    for source_column in ("lead_role_game", "heavy_load_game", "rotation_role_game", "bench_risk_game"):
        feature_prefix = source_column.removesuffix("_game")
        for window in (5, 10):
            engineered_columns[f"{feature_prefix}_rate_{window}"] = (
                ordered.groupby("player_id")[source_column]
                .transform(lambda series: series.shift(1).rolling(window, min_periods=1).mean())
            )

    safe_minutes = ordered["minutes"].replace(0, np.nan)
    safe_turnovers = ordered["turnovers"].replace(0, np.nan)
    safe_team_assists = ordered["team_assists_form_5"].replace(0, np.nan)
    ordered["assists_per_minute_raw"] = ordered["assists"].div(safe_minutes).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ordered["points_per_minute_raw"] = ordered["points"].div(safe_minutes).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ordered["rebounds_per_minute_raw"] = ordered["rebounds"].div(safe_minutes).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ordered["assist_to_turnover_raw"] = ordered["assists"].div(safe_turnovers).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ordered["assist_share_raw"] = ordered["assists"].div(safe_team_assists).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ordered["assist_plus_usage_raw"] = ordered["assists"] + (ordered["usage_rate"] * 10.0)
    ordered["assist_creation_raw"] = ordered["assists_per_minute_raw"] * ordered["usage_rate"]
    ordered["assist_load_raw"] = ordered["assists"] * ordered["minutes"]

    assist_specialty_sources = {
        "assists_per_minute": "assists_per_minute_raw",
        "points_per_minute": "points_per_minute_raw",
        "rebounds_per_minute": "rebounds_per_minute_raw",
        "assist_to_turnover": "assist_to_turnover_raw",
        "assist_share": "assist_share_raw",
        "assist_plus_usage": "assist_plus_usage_raw",
        "assist_creation": "assist_creation_raw",
        "assist_load": "assist_load_raw",
    }
    for feature_name, source_column in assist_specialty_sources.items():
        for window in (3, 5, 10):
            engineered_columns[f"{feature_name}_rolling_{window}"] = (
                ordered.groupby("player_id")[source_column]
                .transform(lambda series: series.shift(1).rolling(window, min_periods=1).mean())
            )
        engineered_columns[f"{feature_name}_season_avg"] = (
            ordered.groupby("player_id")[source_column]
            .transform(lambda series: series.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )
        engineered_columns[f"{feature_name}_last_game"] = ordered.groupby("player_id")[source_column].shift(1)

    engineered = pd.DataFrame(engineered_columns, index=ordered.index)
    engineered["projected_minutes"] = (
        engineered["minutes_rolling_3"] * 0.5
        + engineered["minutes_rolling_5"] * 0.35
        + engineered["minutes_season_avg"] * 0.15
    )
    engineered["projected_minutes_delta"] = engineered["projected_minutes"] - engineered["minutes_season_avg"]
    engineered["role_change_10"] = engineered["lead_role_rate_5"] - engineered["lead_role_rate_10"]
    engineered["bench_risk_delta"] = engineered["bench_risk_rate_5"] - engineered["bench_risk_rate_10"]
    engineered["minutes_usage_interaction"] = engineered["projected_minutes"] * engineered["usage_rate_rolling_5"]
    engineered["projected_opportunity"] = engineered["projected_minutes"] * engineered["usage_rate_season_avg"]
    engineered["rest_advantage"] = ordered["rest_days"] - ordered.groupby("opponent_abbr")["rest_days"].transform("median")
    engineered["pace_edge"] = ordered["team_pace"] - ordered["opp_pace"]
    engineered["defense_edge"] = ordered["opp_def_rating"] - ordered["team_off_rating"]
    engineered["team_form_edge_points"] = ordered["team_points_form_5"] - ordered["opp_points_allowed_5"]
    engineered["team_form_edge_assists"] = ordered["team_assists_form_5"] - ordered["opp_assists_allowed_5"]
    engineered["team_form_edge_rebounds"] = ordered["team_rebounds_form_5"] - ordered["opp_rebounds_allowed_5"]
    engineered["win_pct_edge"] = ordered["team_win_pct_10"] - ordered["opp_win_pct_10"]
    engineered["last_game_points_delta"] = engineered["points_rolling_3"] - engineered["points_last_game"]
    engineered["last_game_assists_delta"] = engineered["assists_rolling_3"] - engineered["assists_last_game"]
    engineered["last_game_rebounds_delta"] = engineered["rebounds_rolling_3"] - engineered["rebounds_last_game"]
    engineered["minutes_trend"] = engineered["minutes_rolling_3"] - engineered["minutes_rolling_10"]
    engineered["usage_trend"] = engineered["usage_rate_rolling_3"] - engineered["usage_rate_rolling_10"]
    engineered["home_minutes_interaction"] = ordered["home"] * engineered["projected_minutes"]
    engineered["back_to_back_usage_penalty"] = ordered["is_back_to_back"] * engineered["usage_rate_rolling_5"]
    engineered["assist_opportunity_index"] = engineered["projected_minutes"] * engineered["assists_per_minute_rolling_5"]
    engineered["assist_creation_index"] = engineered["projected_minutes"] * engineered["assist_creation_rolling_5"]
    engineered["assist_share_delta"] = engineered["assist_share_rolling_5"] - engineered["assist_share_season_avg"]
    engineered["assist_role_boost"] = engineered["lead_role_rate_5"] * engineered["assist_share_rolling_5"]
    engineered["assist_matchup_index"] = ordered["opp_assists_allowed_5"] * engineered["assist_share_rolling_5"]
    engineered["assist_stability_index"] = engineered["assists_rolling_10"] - engineered["assists_rolling_std_10"]
    engineered["playmaking_pressure_index"] = ordered["team_assists_form_5"] - ordered["opp_assists_allowed_5"]

    # Role stability features
    engineered["minutes_volatility_10"] = (
        ordered.groupby("player_id")["minutes"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=2).std())
    )
    engineered["starter_rate_5"] = engineered["starter_rolling_5"]
    engineered["starter_rate_10"] = engineered["starter_rolling_10"]
    safe_minutes_10 = engineered["minutes_rolling_10"].replace(0, np.nan).fillna(1.0)
    engineered["starter_consistency"] = 1.0 - (engineered["minutes_volatility_10"] / safe_minutes_10)
    engineered["role_stability_score"] = engineered["starter_rate_10"] * engineered["starter_consistency"]

    # Defensive involvement
    engineered["defensive_actions_rolling_5"] = engineered["steals_rolling_5"] + engineered["blocks_rolling_5"]
    engineered["defensive_actions_season_avg"] = engineered["steals_season_avg"] + engineered["blocks_season_avg"]

    # Rebound split ratios
    total_reb_5 = engineered["rebounds_rolling_5"].replace(0, np.nan).fillna(1.0)
    engineered["off_reb_share_5"] = engineered["off_rebounds_rolling_5"] / total_reb_5
    engineered["def_reb_share_5"] = engineered["def_rebounds_rolling_5"] / total_reb_5

    # Plus/minus trend
    engineered["plus_minus_trend"] = engineered["plus_minus_rolling_3"] - engineered["plus_minus_rolling_10"]

    # Implied game total: pace × average offensive ratings — strong proxy for scoring environment
    safe_opp_pace = ordered["opp_pace"].replace(0, np.nan).fillna(ordered["team_pace"])
    engineered["implied_game_total"] = (
        (ordered["team_pace"] + safe_opp_pace) / 2
        * (ordered["team_off_rating"] + ordered["opp_off_rating"]) / 200
    )
    engineered["team_pace_edge_ratio"] = ordered["team_pace"] / safe_opp_pace.replace(0, np.nan).fillna(1.0)

    # Opponent-specific history features: how player performs against this specific team
    if "opponent_abbr" in ordered.columns:
        for stat in ("points", "assists", "rebounds", "minutes"):
            vs_opp = (
                ordered.groupby(["player_id", "opponent_abbr"])[stat]
                .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
            )
            season_fallback = (
                ordered.groupby("player_id")[stat]
                .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
            )
            engineered[f"vs_opp_{stat}_avg"] = vs_opp.fillna(season_fallback)
        engineered["vs_opp_games_count"] = (
            ordered.groupby(["player_id", "opponent_abbr"])["points"]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).count())
            .fillna(0.0)
        )
        for stat in ("points", "assists", "rebounds"):
            engineered[f"vs_opp_{stat}_last"] = (
                ordered.groupby(["player_id", "opponent_abbr"])[stat]
                .transform(lambda s: s.shift(1))
                .fillna(0.0)
            )

    ordered = pd.concat([ordered, engineered], axis=1)

    feature_columns = [
        column
        for column in [*PREGAME_CONTEXT_COLUMNS, *engineered.columns.tolist()]
        if column in ordered.columns
    ]
    return ordered[["game_date", "player_id", *TARGET_COLUMNS, "minutes", *feature_columns]]


def _prop_style_metrics(y_true: pd.Series, predictions: pd.Series, tolerances: list[int]) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }
    absolute_errors = (y_true - predictions).abs()
    for tolerance in tolerances:
        metrics[f"within_{tolerance}"] = float((absolute_errors <= tolerance).mean())
    return metrics


def _train_target(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target: str,
) -> tuple[Pipeline, dict[str, float], list[str]]:
    feature_columns = [
        column
        for column in training_frame.columns
        if column not in {"game_date", "player_id", *TARGET_COLUMNS, *AUXILIARY_TARGETS}
    ]
    X_train = training_frame[feature_columns]
    y_train = training_frame[target]
    X_valid = validation_frame[feature_columns]
    y_valid = validation_frame[target]
    days_from_end = (training_frame["game_date"].max() - training_frame["game_date"]).dt.days
    sample_weight = (0.995 ** days_from_end).clip(lower=0.15)
    # Down-weight short-minute games (garbage time / early foul trouble / DNP-adjacent)
    quality_weight = np.where(training_frame["minutes"].fillna(0) < 15, 0.25, 1.0)
    sample_weight = (sample_weight * quality_weight).clip(lower=0.05)

    best_model = None
    best_metrics = None
    best_mae = float("inf")
    for params in TARGET_CONFIG[target]["candidate_params"]:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    LGBMRegressor(
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        **params,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train, regressor__sample_weight=sample_weight)
        predictions = model.predict(X_valid)
        metrics = _prop_style_metrics(y_valid, predictions, TARGET_CONFIG[target]["tolerances"])

        line_column = TARGET_CONFIG[target]["line_column"]
        if line_column in validation_frame.columns and (validation_frame[line_column] > 0).any():
            available = validation_frame[line_column].notna()
            line_values = validation_frame.loc[available, line_column]
            line_predictions = pd.Series(predictions, index=validation_frame.index).loc[available]
            actual = y_valid.loc[available]
            metrics["over_under_hit_rate"] = float(((line_predictions > line_values) == (actual > line_values)).mean())

        if metrics["mae"] < best_mae:
            best_model = model
            best_metrics = metrics
            best_mae = metrics["mae"]

    return best_model, best_metrics, feature_columns


def _train_auxiliary_target(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target: str,
) -> tuple[Pipeline, dict[str, float], list[str]]:
    feature_columns = [
        column
        for column in training_frame.columns
        if column not in {"game_date", "player_id", *TARGET_COLUMNS, *AUXILIARY_TARGETS}
    ]
    X_train = training_frame[feature_columns]
    y_train = training_frame[target]
    X_valid = validation_frame[feature_columns]
    y_valid = validation_frame[target]
    days_from_end = (training_frame["game_date"].max() - training_frame["game_date"]).dt.days
    sample_weight = (0.995 ** days_from_end).clip(lower=0.15)
    quality_weight = np.where(training_frame["minutes"].fillna(0) < 15, 0.25, 1.0)
    sample_weight = (sample_weight * quality_weight).clip(lower=0.05)

    best_model = None
    best_metrics = None
    best_mae = float("inf")
    for params in AUXILIARY_TARGET_CONFIG[target]["candidate_params"]:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    LGBMRegressor(
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        **params,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train, regressor__sample_weight=sample_weight)
        predictions = model.predict(X_valid)
        metrics = _prop_style_metrics(y_valid, predictions, AUXILIARY_TARGET_CONFIG[target]["tolerances"])

        if metrics["mae"] < best_mae:
            best_model = model
            best_metrics = metrics
            best_mae = metrics["mae"]

    return best_model, best_metrics, feature_columns


def train_models(dataset_path: Path, output_dir: Path, validation_ratio: float) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(dataset_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Expected at minimum game_date, player_id, points, assists, rebounds."
        )

    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame = frame.dropna(subset=["game_date"]).copy()
    feature_frame = _build_features(frame)
    feature_frame = feature_frame.dropna(subset=["points", "assists", "rebounds"]).copy()
    feature_frame = feature_frame.sort_values("game_date")

    split_index = int(len(feature_frame) * (1 - validation_ratio))
    training_frame = feature_frame.iloc[:split_index]
    validation_frame = feature_frame.iloc[split_index:]
    if training_frame.empty or validation_frame.empty:
        raise ValueError("Training/validation split produced an empty partition. Use more rows or a smaller validation ratio.")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_target = {}
    metadata = {"targets": {}, "auxiliary": {}}

    artifact_names = {
        "points": "model_points.pkl",
        "assists": "model_assists.pkl",
        "rebounds": "model_rebounds.pkl",
    }
    auxiliary_artifact_names = {
        "minutes": "model_minutes.pkl",
    }

    for target in TARGET_COLUMNS:
        model, metrics, feature_columns = _train_target(training_frame, validation_frame, target)
        artifact_name = artifact_names[target]
        with open(output_dir / artifact_name, "wb") as artifact:
            pickle.dump(model, artifact)

        metrics_by_target[target] = metrics
        metadata["targets"][target] = {
            "artifact_name": artifact_name,
            "feature_names": feature_columns,
            "metrics": metrics,
        }

    for target in AUXILIARY_TARGETS:
        model, metrics, feature_columns = _train_auxiliary_target(training_frame, validation_frame, target)
        artifact_name = auxiliary_artifact_names[target]
        with open(output_dir / artifact_name, "wb") as artifact:
            pickle.dump(model, artifact)

        metadata["auxiliary"][target] = {
            "artifact_name": artifact_name,
            "feature_names": feature_columns,
            "metrics": metrics,
        }

    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metrics_by_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NBA player stat regression models.")
    parser.add_argument("--dataset", required=True, help="CSV file with game-by-game player stats.")
    parser.add_argument("--output-dir", default=".", help="Directory to write model artifacts.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Fraction of rows reserved for validation.")
    args = parser.parse_args()

    metrics_by_target = train_models(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        validation_ratio=args.validation_ratio,
    )

    print(json.dumps(metrics_by_target, indent=2))


if __name__ == "__main__":
    main()
