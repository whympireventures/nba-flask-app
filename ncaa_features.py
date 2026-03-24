from __future__ import annotations

from datetime import datetime
from statistics import pstdev
from typing import Any


INFERENCE_STAT_SOURCES = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "minutes": "minutes",
    "fg_pct": "fg_pct",
    "three_pt_pct": "three_pt_pct",
    "turnovers": "turnovers",
}


def _safe_float(value: Any) -> float:
    if value in (None, "", "None", "--"):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "")
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        try:
            return float(minutes) + (float(seconds) / 60.0)
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_percentage(value: Any) -> float:
    number = _safe_float(value)
    if 0.0 < number <= 1.0:
        return number * 100.0
    return number


def _parse_game_datetime(game: dict[str, Any]) -> datetime:
    raw_candidates = [
        game.get("game", {}).get("date"),
        game.get("date"),
    ]
    for raw in raw_candidates:
        if not raw:
            continue
        normalized = str(raw).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%b %d, %Y"):
                try:
                    return datetime.strptime(str(raw), fmt)
                except ValueError:
                    continue
    return datetime.min


def _normalize_game(game: dict[str, Any]) -> dict[str, float]:
    normalized = {
        "points": _safe_float(game.get("points")),
        "rebounds": _safe_float(game.get("totReb", game.get("rebounds"))),
        "assists": _safe_float(game.get("assists")),
        "minutes": _safe_float(game.get("min", game.get("minutes"))),
        "fg_pct": _normalize_percentage(game.get("fgp", game.get("fg_pct"))),
        "three_pt_pct": _normalize_percentage(game.get("tpp", game.get("three_pt_pct"))),
        "turnovers": _safe_float(game.get("turnovers")),
    }
    normalized["game_id"] = int(_safe_float(game.get("game", {}).get("id")))
    normalized["game_datetime"] = _parse_game_datetime(game)
    normalized["opponent"] = str(
        game.get("opponent_name")
        or game.get("opponent_abbr")
        or ""
    ).strip().upper()
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


def build_feature_row(
    game_logs: list[dict[str, Any]],
    upcoming_context: dict[str, Any] | None = None,
) -> dict[str, float]:
    ordered_games = sort_games(game_logs)
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
        feature_row[f"lead_role_rate_{window}"] = _window_rate(games, lambda game: game.get("minutes", 0.0) >= 32.0)
        feature_row[f"heavy_load_rate_{window}"] = _window_rate(games, lambda game: game.get("minutes", 0.0) >= 36.0)
        feature_row[f"rotation_role_rate_{window}"] = _window_rate(games, lambda game: game.get("minutes", 0.0) >= 24.0)
        feature_row[f"bench_risk_rate_{window}"] = _window_rate(games, lambda game: game.get("minutes", 0.0) < 18.0)

    derived_sources = {
        "points_per_minute": lambda game: game.get("points", 0.0) / max(game.get("minutes", 0.0), 1.0),
        "rebounds_per_minute": lambda game: game.get("rebounds", 0.0) / max(game.get("minutes", 0.0), 1.0),
        "assists_per_minute": lambda game: game.get("assists", 0.0) / max(game.get("minutes", 0.0), 1.0),
        "turnovers_per_minute": lambda game: game.get("turnovers", 0.0) / max(game.get("minutes", 0.0), 1.0),
        "assist_to_turnover": lambda game: game.get("assists", 0.0) / max(game.get("turnovers", 0.0), 1.0),
        "shooting_efficiency": lambda game: (game.get("fg_pct", 0.0) * 0.7) + (game.get("three_pt_pct", 0.0) * 0.3),
    }
    for feature_name, value_getter in derived_sources.items():
        derived_games = [{feature_name: value_getter(game)} for game in ordered_games]
        for window in (3, 5, 10):
            games = derived_games[:window]
            feature_row[f"{feature_name}_rolling_{window}"] = _window_mean(games, feature_name)
        feature_row[f"{feature_name}_season_avg"] = _window_mean(derived_games, feature_name)
        feature_row[f"{feature_name}_last_game"] = derived_games[0].get(feature_name, 0.0) if derived_games else 0.0

    feature_row["projected_minutes"] = (
        feature_row["minutes_rolling_3"] * 0.5
        + feature_row["minutes_rolling_5"] * 0.35
        + feature_row["minutes_season_avg"] * 0.15
    )
    feature_row["projected_minutes_delta"] = feature_row["projected_minutes"] - feature_row["minutes_season_avg"]
    feature_row["minutes_trend"] = feature_row["minutes_rolling_3"] - feature_row["minutes_rolling_10"]
    feature_row["points_trend"] = feature_row["points_rolling_3"] - feature_row["points_rolling_10"]
    feature_row["rebounds_trend"] = feature_row["rebounds_rolling_3"] - feature_row["rebounds_rolling_10"]
    feature_row["assists_trend"] = feature_row["assists_rolling_3"] - feature_row["assists_rolling_10"]
    feature_row["turnovers_trend"] = feature_row["turnovers_rolling_3"] - feature_row["turnovers_rolling_10"]
    feature_row["role_change_10"] = feature_row["lead_role_rate_5"] - feature_row["lead_role_rate_10"]
    feature_row["bench_risk_delta"] = feature_row["bench_risk_rate_5"] - feature_row["bench_risk_rate_10"]
    feature_row["points_workload_index"] = feature_row["projected_minutes"] * feature_row["points_per_minute_rolling_5"]
    feature_row["rebounds_workload_index"] = feature_row["projected_minutes"] * feature_row["rebounds_per_minute_rolling_5"]
    feature_row["assists_workload_index"] = feature_row["projected_minutes"] * feature_row["assists_per_minute_rolling_5"]

    opponent = str(
        (upcoming_context or {}).get("opponent")
        or (upcoming_context or {}).get("opponent_abbr")
        or ""
    ).strip().upper()
    if opponent and ordered_games:
        opponent_games = [game for game in ordered_games if game.get("opponent") == opponent]
    else:
        opponent_games = []

    for stat in INFERENCE_STAT_SOURCES:
        source_key = INFERENCE_STAT_SOURCES[stat]
        if opponent_games:
            feature_row[f"vs_opp_{stat}_avg"] = _window_mean(opponent_games, source_key)
            feature_row[f"vs_opp_{stat}_last"] = opponent_games[0].get(source_key, 0.0)
        else:
            feature_row[f"vs_opp_{stat}_avg"] = feature_row.get(f"{stat}_season_avg", 0.0)
            feature_row[f"vs_opp_{stat}_last"] = 0.0
    feature_row["vs_opp_games_count"] = float(len(opponent_games))

    return feature_row
