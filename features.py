from __future__ import annotations

from datetime import datetime
from statistics import pstdev
from typing import Any

import pandas as pd


LEGACY_FEATURE_ORDER = [
    "points",
    "fgm",
    "fga",
    "fgp",
    "ftp",
    "tpm",
    "tpa",
    "tpp",
    "offReb",
    "defReb",
    "totReb",
    "assists",
    "pFouls",
    "steals",
    "turnovers",
    "blocks",
    "plusMinus",
]

INFERENCE_STAT_SOURCES = {
    "points": "points",
    "assists": "assists",
    "rebounds": "totReb",
    "minutes": "minutes",
    "fg_pct": "fgp",
    "three_pt_pct": "tpp",
    "ft_pct": "ftp",
    "usage_rate": "usage_rate",
    "true_shooting_pct": "true_shooting_pct",
    "player_pace": "player_pace",
    "opp_def_rating": "opp_def_rating",
    "opp_pace": "opp_pace",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "off_rebounds": "offReb",
    "def_rebounds": "defReb",
    "plus_minus": "plusMinus",
    "player_off_rating": "player_off_rating",
    "player_def_rating": "player_def_rating",
    "starter": "starter",
}

CONTEXT_DEFAULTS = {
    "home": 0.0,
    "rest_days": 3.0,
    "is_back_to_back": 0.0,
    "starter": 1.0,
    "usage_rate": 0.0,
    "true_shooting_pct": 0.0,
    "player_pace": 0.0,
    "player_off_rating": 0.0,
    "player_def_rating": 0.0,
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


def _safe_float(value: Any) -> float:
    if value in (None, "", "None"):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + (float(seconds) / 60.0)
    return float(text)


def _parse_game_datetime(game: dict[str, Any]) -> datetime:
    raw_candidates = [
        game.get("game", {}).get("date"),
        game.get("date", {}).get("start") if isinstance(game.get("date"), dict) else None,
        game.get("date"),
    ]
    for raw in raw_candidates:
        if not raw:
            continue
        normalized = str(raw).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            try:
                return datetime.strptime(str(raw), "%b %d, %Y")
            except ValueError:
                continue
    return datetime.min


def _normalize_game(game: dict[str, Any]) -> dict[str, float]:
    normalized = {
        "points": _safe_float(game.get("points")),
        "fgm": _safe_float(game.get("fgm")),
        "fga": _safe_float(game.get("fga")),
        "fgp": _safe_float(game.get("fgp")),
        "ftp": _safe_float(game.get("ftp")),
        "tpm": _safe_float(game.get("tpm")),
        "tpa": _safe_float(game.get("tpa")),
        "tpp": _safe_float(game.get("tpp")),
        "offReb": _safe_float(game.get("offReb")),
        "defReb": _safe_float(game.get("defReb")),
        "totReb": _safe_float(game.get("totReb")),
        "assists": _safe_float(game.get("assists")),
        "pFouls": _safe_float(game.get("pFouls")),
        "steals": _safe_float(game.get("steals")),
        "turnovers": _safe_float(game.get("turnovers")),
        "blocks": _safe_float(game.get("blocks")),
        "plusMinus": _safe_float(game.get("plusMinus")),
        "minutes": _safe_float(game.get("min")),
        "usage_rate": _safe_float(game.get("usage_rate")),
        "true_shooting_pct": _safe_float(game.get("true_shooting_pct")),
        "player_pace": _safe_float(game.get("player_pace")),
        "player_off_rating": _safe_float(game.get("player_off_rating")),
        "player_def_rating": _safe_float(game.get("player_def_rating")),
        "starter": _safe_float(game.get("starter")),
    }
    normalized["game_id"] = int(_safe_float(game.get("game", {}).get("id")))
    normalized["game_datetime"] = _parse_game_datetime(game)
    normalized["opponent_abbr"] = str(game.get("opponent_abbr", "") or "").upper()
    return normalized


def sort_games(game_logs: list[dict[str, Any]]) -> list[dict[str, float]]:
    normalized_games = [_normalize_game(game) for game in game_logs]
    return sorted(
        normalized_games,
        key=lambda game: (game["game_datetime"], game["game_id"]),
        reverse=True,
    )


def _window_mean(games: list[dict[str, float]], stat_key: str) -> float:
    if not games:
        return 0.0
    values = [game.get(stat_key, 0.0) for game in games]
    return sum(values) / len(values)


def _window_std(games: list[dict[str, float]], stat_key: str) -> float:
    if len(games) <= 1:
        return 0.0
    values = [game.get(stat_key, 0.0) for game in games]
    return pstdev(values)


def _window_rate(games: list[dict[str, float]], predicate) -> float:
    if not games:
        return 0.0
    return sum(1.0 for game in games if predicate(game)) / len(games)


def _build_inference_context(
    ordered_games: list[dict[str, float]],
    upcoming_context: dict[str, Any] | None,
) -> dict[str, float]:
    context = {key: float(value) for key, value in CONTEXT_DEFAULTS.items()}
    provided_keys: set[str] = set()
    if upcoming_context:
        for key, value in upcoming_context.items():
            if key in context and value not in (None, ""):
                context[key] = float(value)
                provided_keys.add(key)

    if ordered_games:
        recent_games = ordered_games[:10]
        if "starter" not in provided_keys:
            context["starter"] = float(_window_mean(recent_games, "minutes") >= 28.0)
        if "usage_rate" not in provided_keys:
            context["usage_rate"] = _window_mean(recent_games, "usage_rate")
        if "true_shooting_pct" not in provided_keys:
            context["true_shooting_pct"] = _window_mean(recent_games, "true_shooting_pct")
        if "player_pace" not in provided_keys:
            context["player_pace"] = _window_mean(recent_games, "player_pace")
        if "player_off_rating" not in provided_keys:
            context["player_off_rating"] = _window_mean(recent_games, "player_off_rating")
        if "player_def_rating" not in provided_keys:
            context["player_def_rating"] = _window_mean(recent_games, "player_def_rating")
    return context


def build_feature_row(game_logs: list[dict[str, Any]], upcoming_context: dict[str, Any] | None = None) -> dict[str, float]:
    if not game_logs:
        raise ValueError("game_logs must contain at least one game.")
    ordered_games = sort_games(game_logs)
    context = _build_inference_context(ordered_games, upcoming_context)
    feature_row: dict[str, float] = {}

    for feature_name, stat_key in INFERENCE_STAT_SOURCES.items():
        for window in (3, 5, 10):
            games = ordered_games[:window]
            feature_row[f"{feature_name}_rolling_{window}"] = _window_mean(games, stat_key)
            feature_row[f"{feature_name}_rolling_std_{window}"] = _window_std(games, stat_key)
        feature_row[f"{feature_name}_season_avg"] = _window_mean(ordered_games, stat_key)
        feature_row[f"{feature_name}_last_game"] = ordered_games[0].get(stat_key, 0.0) if ordered_games else 0.0

    for window in (5, 10):
        games = ordered_games[:window]
        feature_row[f"lead_role_rate_{window}"] = _window_rate(games, lambda game: game.get("minutes", 0.0) >= 30.0)
        feature_row[f"heavy_load_rate_{window}"] = _window_rate(games, lambda game: game.get("minutes", 0.0) >= 34.0)
        feature_row[f"rotation_role_rate_{window}"] = _window_rate(games, lambda game: game.get("minutes", 0.0) >= 24.0)
        feature_row[f"bench_risk_rate_{window}"] = _window_rate(games, lambda game: game.get("minutes", 0.0) < 20.0)

    assist_specialty_builders = {
        "assists_per_minute": lambda game: game.get("assists", 0.0) / max(game.get("minutes", 0.0), 1.0),
        "points_per_minute": lambda game: game.get("points", 0.0) / max(game.get("minutes", 0.0), 1.0),
        "rebounds_per_minute": lambda game: game.get("totReb", 0.0) / max(game.get("minutes", 0.0), 1.0),
        "assist_to_turnover": lambda game: game.get("assists", 0.0) / max(game.get("turnovers", 0.0), 1.0),
        "assist_plus_usage": lambda game: game.get("assists", 0.0) + (game.get("usage_rate", 0.0) * 10.0),
        "assist_creation": lambda game: (game.get("assists", 0.0) / max(game.get("minutes", 0.0), 1.0)) * game.get("usage_rate", 0.0),
        "assist_load": lambda game: game.get("assists", 0.0) * game.get("minutes", 0.0),
    }
    for feature_name, value_getter in assist_specialty_builders.items():
        derived_games = [{feature_name: value_getter(game)} for game in ordered_games]
        for window in (3, 5, 10):
            games = derived_games[:window]
            feature_row[f"{feature_name}_rolling_{window}"] = _window_mean(games, feature_name)
        feature_row[f"{feature_name}_season_avg"] = _window_mean(derived_games, feature_name)
        feature_row[f"{feature_name}_last_game"] = derived_games[0].get(feature_name, 0.0) if derived_games else 0.0

    for window in (3, 5, 10):
        assists_games = ordered_games[:window]
        team_assists = max(context["team_assists_form_5"], 1.0)
        feature_row[f"assist_share_rolling_{window}"] = _window_mean(assists_games, "assists") / team_assists if assists_games else 0.0
    feature_row["assist_share_season_avg"] = feature_row["assist_share_rolling_10"]
    feature_row["assist_share_last_game"] = feature_row["assists_last_game"] / max(context["team_assists_form_5"], 1.0)

    feature_row.update(context)
    advanced_fallbacks = {
        "usage_rate": context["usage_rate"],
        "true_shooting_pct": context["true_shooting_pct"],
        "player_pace": context["player_pace"],
    }
    for feature_name, fallback_value in advanced_fallbacks.items():
        if fallback_value == 0.0:
            continue
        for suffix in ("rolling_3", "rolling_5", "rolling_10", "season_avg", "last_game"):
            column_name = f"{feature_name}_{suffix}"
            if feature_row.get(column_name, 0.0) == 0.0:
                feature_row[column_name] = fallback_value

    feature_row["projected_minutes"] = (
        feature_row["minutes_rolling_3"] * 0.5
        + feature_row["minutes_rolling_5"] * 0.35
        + feature_row["minutes_season_avg"] * 0.15
    )
    feature_row["projected_minutes_delta"] = feature_row["projected_minutes"] - feature_row["minutes_season_avg"]
    feature_row["role_change_10"] = feature_row["lead_role_rate_5"] - feature_row["lead_role_rate_10"]
    feature_row["bench_risk_delta"] = feature_row["bench_risk_rate_5"] - feature_row["bench_risk_rate_10"]
    feature_row["minutes_usage_interaction"] = feature_row["projected_minutes"] * feature_row["usage_rate_rolling_5"]
    feature_row["projected_opportunity"] = feature_row["projected_minutes"] * feature_row["usage_rate_season_avg"]
    feature_row["rest_advantage"] = context["rest_days"]
    feature_row["pace_edge"] = context["team_pace"] - context["opp_pace"]
    feature_row["defense_edge"] = context["opp_def_rating"] - context["team_off_rating"]
    feature_row["team_form_edge_points"] = context["team_points_form_5"] - context["opp_points_allowed_5"]
    feature_row["team_form_edge_assists"] = context["team_assists_form_5"] - context["opp_assists_allowed_5"]
    feature_row["team_form_edge_rebounds"] = context["team_rebounds_form_5"] - context["opp_rebounds_allowed_5"]
    feature_row["win_pct_edge"] = context["team_win_pct_10"] - context["opp_win_pct_10"]
    feature_row["last_game_points_delta"] = feature_row["points_rolling_3"] - feature_row["points_last_game"]
    feature_row["last_game_assists_delta"] = feature_row["assists_rolling_3"] - feature_row["assists_last_game"]
    feature_row["last_game_rebounds_delta"] = feature_row["rebounds_rolling_3"] - feature_row["rebounds_last_game"]
    feature_row["minutes_trend"] = feature_row["minutes_rolling_3"] - feature_row["minutes_rolling_10"]
    feature_row["usage_trend"] = feature_row["usage_rate_rolling_3"] - feature_row["usage_rate_rolling_10"]
    feature_row["home_minutes_interaction"] = context["home"] * feature_row["projected_minutes"]
    feature_row["back_to_back_usage_penalty"] = context["is_back_to_back"] * feature_row["usage_rate_rolling_5"]
    feature_row["assist_opportunity_index"] = feature_row["projected_minutes"] * feature_row["assists_per_minute_rolling_5"]
    feature_row["assist_creation_index"] = feature_row["projected_minutes"] * feature_row["assist_creation_rolling_5"]
    feature_row["assist_share_delta"] = feature_row["assist_share_rolling_5"] - feature_row["assist_share_season_avg"]
    feature_row["assist_role_boost"] = feature_row["lead_role_rate_5"] * feature_row["assist_share_rolling_5"]
    feature_row["assist_matchup_index"] = context["opp_assists_allowed_5"] * feature_row["assist_share_rolling_5"]
    feature_row["assist_stability_index"] = feature_row["assists_rolling_10"] - feature_row["assists_rolling_std_10"]
    feature_row["playmaking_pressure_index"] = context["team_assists_form_5"] - context["opp_assists_allowed_5"]

    # Implied game total: pace × average offensive ratings — scoring environment proxy
    _team_pace = context.get("team_pace", 0.0)
    _opp_pace = context.get("opp_pace", 0.0) or _team_pace
    _team_off = context.get("team_off_rating", 0.0)
    _opp_off = context.get("opp_off_rating", 0.0)
    if _team_pace > 0 and _team_off > 0 and _opp_off > 0:
        feature_row["implied_game_total"] = (_team_pace + _opp_pace) / 2 * (_team_off + _opp_off) / 200
        feature_row["team_pace_edge_ratio"] = _team_pace / max(_opp_pace, 1.0)
    else:
        feature_row["implied_game_total"] = 0.0
        feature_row["team_pace_edge_ratio"] = 1.0

    # Opponent-specific matchup history features
    opponent_abbr_for_matchup = str((upcoming_context or {}).get("opponent_abbr", "") or "").upper()
    if opponent_abbr_for_matchup and ordered_games:
        opp_games = [g for g in ordered_games if g.get("opponent_abbr", "").upper() == opponent_abbr_for_matchup]
    else:
        opp_games = []
    for stat, key in [("points", "points"), ("assists", "assists"), ("rebounds", "totReb")]:
        if opp_games:
            feature_row[f"vs_opp_{stat}_avg"] = _window_mean(opp_games, key)
            feature_row[f"vs_opp_{stat}_last"] = opp_games[0].get(key, 0.0)
        else:
            feature_row[f"vs_opp_{stat}_avg"] = feature_row.get(f"{stat}_season_avg", 0.0)
            feature_row[f"vs_opp_{stat}_last"] = 0.0
    feature_row["vs_opp_minutes_avg"] = _window_mean(opp_games, "minutes") if opp_games else feature_row.get("minutes_season_avg", 0.0)
    feature_row["vs_opp_games_count"] = float(len(opp_games))

    # Role stability: how consistent are minutes over last 10 games
    recent_10 = ordered_games[:10]
    feature_row["minutes_volatility_10"] = _window_std(recent_10, "minutes")
    feature_row["starter_rate_5"] = _window_mean(ordered_games[:5], "starter")
    feature_row["starter_rate_10"] = _window_mean(recent_10, "starter")
    feature_row["starter_consistency"] = 1.0 - (feature_row["minutes_volatility_10"] / max(feature_row["minutes_rolling_10"], 1.0))
    feature_row["role_stability_score"] = feature_row["starter_rate_10"] * feature_row["starter_consistency"]

    # Defensive involvement: steals + blocks per game
    feature_row["defensive_actions_rolling_5"] = feature_row["steals_rolling_5"] + feature_row["blocks_rolling_5"]
    feature_row["defensive_actions_season_avg"] = feature_row["steals_season_avg"] + feature_row["blocks_season_avg"]

    # Rebound split ratio: off vs total rebounds
    total_reb_5 = max(feature_row["rebounds_rolling_5"], 1.0)
    feature_row["off_reb_share_5"] = feature_row["off_rebounds_rolling_5"] / total_reb_5
    feature_row["def_reb_share_5"] = feature_row["def_rebounds_rolling_5"] / total_reb_5

    # Net efficiency: plus_minus trend
    feature_row["plus_minus_trend"] = feature_row["plus_minus_rolling_3"] - feature_row["plus_minus_rolling_10"]

    return feature_row


def build_legacy_feature_frame(game_logs: list[dict[str, Any]]) -> pd.DataFrame:
    ordered_games = sort_games(game_logs)[:5]
    if not ordered_games:
        row = {f"Column_{index}": 0.0 for index, _ in enumerate(LEGACY_FEATURE_ORDER)}
        return pd.DataFrame([row])

    averaged = {}
    for index, stat in enumerate(LEGACY_FEATURE_ORDER):
        averaged[f"Column_{index}"] = sum(game[stat] for game in ordered_games) / len(ordered_games)
    return pd.DataFrame([averaged])
