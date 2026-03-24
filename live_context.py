from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from api_client import _season_player_dashboard, _recent_player_dashboard
from config import settings

try:
    from injury_client import fetch_injury_report, get_out_player_names, get_player_status
    _INJURY_CLIENT_AVAILABLE = True
except ImportError:
    _INJURY_CLIENT_AVAILABLE = False

try:
    from espn_game_client import get_player_game_status
    _ESPN_GAME_CLIENT_AVAILABLE = True
except ImportError:
    _ESPN_GAME_CLIENT_AVAILABLE = False

try:
    from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
    from nba_api.stats.static import teams as nba_static_teams
except ImportError:  # pragma: no cover
    leaguedashplayerstats = None
    leaguedashteamstats = None
    nba_static_teams = None


def _normalize_date(raw_date: str | None) -> datetime | None:
    if not raw_date:
        return None
    try:
        return datetime.fromisoformat(raw_date)
    except ValueError:
        try:
            return datetime.strptime(raw_date, "%b %d, %Y")
        except ValueError:
            return None


@lru_cache(maxsize=8)
def _team_context_by_abbr(season_start_year: int) -> dict[str, dict[str, float]]:
    if leaguedashteamstats is None:
        return {}

    season = f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
    try:
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        frame = endpoint.get_data_frames()[0]
    except Exception:
        return {}

    team_id_to_abbr = {}
    if nba_static_teams is not None:
        team_id_to_abbr = {
            int(team["id"]): str(team["abbreviation"]).upper()
            for team in nba_static_teams.get_teams()
        }

    context = {}
    for _, row in frame.iterrows():
        team_abbr = team_id_to_abbr.get(int(row["TEAM_ID"])) if row.get("TEAM_ID") is not None else None
        if not team_abbr:
            continue
        context[team_abbr] = {
            "team_pace": float(row.get("PACE", 0.0)),
            "team_off_rating": float(row.get("OFF_RATING", 0.0)),
            "team_def_rating": float(row.get("DEF_RATING", 0.0)),
        }
    return context


@lru_cache(maxsize=8)
def _player_context_by_id(season_start_year: int) -> dict[int, dict[str, float]]:
    if leaguedashplayerstats is None:
        return {}

    season = f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
    try:
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        frame = endpoint.get_data_frames()[0]
    except Exception:
        return {}

    context = {}
    for _, row in frame.iterrows():
        player_id = row.get("PLAYER_ID")
        if player_id is None:
            continue
        context[int(player_id)] = {
            "usage_rate": float(row.get("USG_PCT", 0.0)),
            "true_shooting_pct": float(row.get("TS_PCT", 0.0)),
            "player_pace": float(row.get("PACE", 0.0)),
            "player_off_rating": float(row.get("OFF_RATING", 0.0)),
            "player_def_rating": float(row.get("DEF_RATING", 0.0)),
        }
    return context


@lru_cache(maxsize=2)
def _team_trends_by_abbr(season_start_year: int) -> dict[str, dict[str, float]]:
    dataset_path = Path("data/player_game_logs.csv")
    if not dataset_path.exists():
        return {}

    try:
        frame = pd.read_csv(
            dataset_path,
            usecols=["season", "game_id", "game_date", "team_abbr", "opponent_abbr", "points", "assists", "rebounds", "win"],
        )
    except Exception:
        return {}

    season = f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
    frame = frame[frame["season"] == season].copy()
    if frame.empty:
        return {}

    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
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

    context = {}
    for team_abbr, team_frame in team_games.groupby("team_abbr"):
        ordered = team_frame.sort_values(["game_date", "game_id"]).tail(10)
        if ordered.empty:
            continue
        recent_five = ordered.tail(5)
        context[str(team_abbr).upper()] = {
            "team_points_form_5": float(recent_five["team_points"].mean()),
            "team_assists_form_5": float(recent_five["team_assists"].mean()),
            "team_rebounds_form_5": float(recent_five["team_rebounds"].mean()),
            "team_points_allowed_5": float(recent_five["team_points_allowed"].mean()),
            "team_assists_allowed_5": float(recent_five["team_assists_allowed"].mean()),
            "team_rebounds_allowed_5": float(recent_five["team_rebounds_allowed"].mean()),
            "team_win_pct_10": float(ordered["team_win"].mean()),
        }
    return context


def _compute_teammate_opportunity(
    team_abbr: str,
    player_id: int | str,
    season_start_year: int,
) -> dict[str, float]:
    """
    Estimates opportunity boost when teammates are unavailable.
    Uses ESPN injury report (real Out/Doubtful data) as primary signal,
    falls back to rolling-minutes proxy when injury data is unavailable.
    """
    if leaguedashplayerstats is None:
        return {"minutes_opportunity_factor": 1.0, "teammate_availability": 1.0}

    try:
        season_rotation = [
            row
            for row in _season_player_dashboard(season_start_year)
            if str(row.get("TEAM_ABBREVIATION", "")).upper() == team_abbr.upper()
            and row.get("PLAYER_ID") is not None
            and int(row.get("GP", 0)) >= 5
        ]
        if not season_rotation:
            return {"minutes_opportunity_factor": 1.0, "teammate_availability": 1.0}

        top_rotation = sorted(season_rotation, key=lambda r: float(r.get("MIN", 0)), reverse=True)[:9]
        target_id = int(player_id) if str(player_id).isdigit() else -1
        teammates = [r for r in top_rotation if int(r["PLAYER_ID"]) != target_id]
        season_teammate_minutes = sum(float(r.get("MIN", 0)) for r in teammates)

        if season_teammate_minutes <= 0:
            return {"minutes_opportunity_factor": 1.0, "teammate_availability": 1.0}

        # --- Primary: ESPN injury report ---
        injury_missing_minutes = 0.0
        used_injury_data = False
        if _INJURY_CLIENT_AVAILABLE:
            try:
                out_names = get_out_player_names(team_abbr)
                if out_names:
                    from nba_api.stats.static import players as _nba_players
                    static_by_id = {int(p["id"]): p["full_name"].lower() for p in _nba_players.get_players()}
                    for r in teammates:
                        pid = int(r["PLAYER_ID"])
                        name = static_by_id.get(pid, "").lower()
                        if name and any(out in name or name in out for out in out_names):
                            injury_missing_minutes += float(r.get("MIN", 0))
                    used_injury_data = True
            except Exception:
                pass

        if used_injury_data and season_teammate_minutes > 0:
            missing_fraction = min(1.0, injury_missing_minutes / season_teammate_minutes)
        else:
            # Fallback: rolling-minutes proxy
            recent_by_id = {
                int(row["PLAYER_ID"]): float(row.get("MIN", 0))
                for row in _recent_player_dashboard(season_start_year, last_n_games=3)
                if row.get("PLAYER_ID")
                and str(row.get("TEAM_ABBREVIATION", "")).upper() == team_abbr.upper()
            }
            recent_teammate_minutes = sum(recent_by_id.get(int(r["PLAYER_ID"]), 0.0) for r in teammates)
            teammate_availability = min(1.0, recent_teammate_minutes / season_teammate_minutes)
            missing_fraction = max(0.0, 1.0 - teammate_availability)

        minutes_opportunity_factor = min(1.25, 1.0 + missing_fraction * 0.5)
        teammate_availability = max(0.0, 1.0 - missing_fraction)

        return {
            "minutes_opportunity_factor": float(minutes_opportunity_factor),
            "teammate_availability": float(teammate_availability),
        }
    except Exception:
        return {"minutes_opportunity_factor": 1.0, "teammate_availability": 1.0}


def build_upcoming_context(
    game_logs: list[dict[str, Any]],
    opponent_abbr: str | None = None,
    game_date: str | None = None,
    player_id: str | None = None,
    home: bool | None = None,
) -> dict[str, float]:
    opponent_abbr = (opponent_abbr or "").strip().upper()
    parsed_game_date = _normalize_date(game_date)
    context = {
        "home": 0.0,
        "rest_days": 3.0,
        "is_back_to_back": 0.0,
        "starter": 1.0,
        "team_pace": 0.0,
        "team_off_rating": 0.0,
        "team_def_rating": 0.0,
        "opp_pace": 0.0,
        "opp_off_rating": 0.0,
        "opp_def_rating": 0.0,
        "usage_rate": 0.0,
        "true_shooting_pct": 0.0,
        "player_pace": 0.0,
        "player_off_rating": 0.0,
        "player_def_rating": 0.0,
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

    if game_logs:
        latest_game = max(
            game_logs,
            key=lambda game: _normalize_date(
                str((game.get("game", {}) or {}).get("date") or game.get("date", "")).replace("Z", "+00:00")
            ) or datetime.min,
        )
        latest_date_raw = latest_game.get("game", {}).get("date") or latest_game.get("date")
        latest_date = _normalize_date(str(latest_date_raw).replace("Z", "+00:00")) if latest_date_raw else None
        latest_team = latest_game.get("team", {}) if isinstance(latest_game.get("team"), dict) else {}

        if latest_date and parsed_game_date:
            rest_days = max((parsed_game_date.date() - latest_date.date()).days - 1, 0)
            context["rest_days"] = float(rest_days)
            context["is_back_to_back"] = float(rest_days == 0)

        if home is not None:
            context["home"] = float(home)

        team_abbr = str(latest_team.get("code", "")).upper()
        team_context = _team_context_by_abbr(settings.season_start_year)
        if team_abbr in team_context:
            context.update(team_context[team_abbr])
        if opponent_abbr in team_context:
            opponent_context = team_context[opponent_abbr]
            context["opp_pace"] = opponent_context.get("team_pace", 0.0)
            context["opp_off_rating"] = opponent_context.get("team_off_rating", 0.0)
            context["opp_def_rating"] = opponent_context.get("team_def_rating", 0.0)

        if player_id and str(player_id).isdigit():
            player_context = _player_context_by_id(settings.season_start_year)
            context.update(player_context.get(int(player_id), {}))

        team_trends = _team_trends_by_abbr(settings.season_start_year)
        if team_abbr in team_trends:
            context.update(team_trends[team_abbr])
        if opponent_abbr in team_trends:
            opponent_trend = team_trends[opponent_abbr]
            context["opp_points_form_5"] = opponent_trend.get("team_points_form_5", 0.0)
            context["opp_assists_form_5"] = opponent_trend.get("team_assists_form_5", 0.0)
            context["opp_rebounds_form_5"] = opponent_trend.get("team_rebounds_form_5", 0.0)
            context["opp_points_allowed_5"] = opponent_trend.get("team_points_allowed_5", 0.0)
            context["opp_assists_allowed_5"] = opponent_trend.get("team_assists_allowed_5", 0.0)
            context["opp_rebounds_allowed_5"] = opponent_trend.get("team_rebounds_allowed_5", 0.0)
            context["opp_win_pct_10"] = opponent_trend.get("team_win_pct_10", 0.0)

    # Pass opponent abbreviation through so build_feature_row can filter matchup history
    context["opponent_abbr"] = opponent_abbr

    # Teammate availability: detect when key rotation players are missing
    team_abbr_for_opportunity = str(latest_team.get("code", "")).upper() if game_logs else ""
    if team_abbr_for_opportunity and player_id:
        context.update(_compute_teammate_opportunity(
            team_abbr_for_opportunity, player_id, settings.season_start_year
        ))
    else:
        context["minutes_opportunity_factor"] = 1.0
        context["teammate_availability"] = 1.0

    # Game-day status: live scoreboard + confirmed starters
    context["game_status"] = "unknown"
    context["game_status_detail"] = ""
    context["confirmed_starter"] = -1.0  # -1 = unknown, 1 = confirmed starter, 0 = confirmed bench
    if _ESPN_GAME_CLIENT_AVAILABLE and team_abbr_for_opportunity and game_logs:
        try:
            player_first = str((game_logs[0].get("player") or {}).get("firstname") or "").strip()
            player_last = str((game_logs[0].get("player") or {}).get("lastname") or "").strip()
            player_full_name = f"{player_first} {player_last}".strip()
            if player_full_name:
                game_day = get_player_game_status(player_full_name, team_abbr_for_opportunity)
                context["game_status"] = game_day["game_status"]
                context["game_status_detail"] = game_day["status_detail"]
                if game_day["confirmed_starter"] is True:
                    context["confirmed_starter"] = 1.0
                elif game_day["confirmed_starter"] is False:
                    context["confirmed_starter"] = 0.0
                # Override home flag if ESPN confirms it
                if game_day["is_home"] is not None and home is None:
                    context["home"] = float(game_day["is_home"])
        except Exception as exc:
            print(f"ESPN game status error: {exc}")

    return context
