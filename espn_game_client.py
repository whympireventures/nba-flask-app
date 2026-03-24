"""
ESPN game-day client.
- Live scoreboard: today's game status (scheduled/live/final/postponed)
- Pre-game lineup: confirmed starters from ESPN game summary (~1 hour pre-tip)

No API key required.
"""
from __future__ import annotations

import time
import requests
from typing import Any

_SCOREBOARD_TTL_LIVE = 30    # 30s when games are in progress
_SCOREBOARD_TTL_IDLE = 300   # 5 min otherwise
_LINEUP_TTL = 300

_scoreboard_cache: list[dict] = []
_scoreboard_cache_time: float = 0.0

_lineup_cache: dict[str, dict[str, bool]] = {}
_lineup_cache_times: dict[str, float] = {}

_boxscore_cache: dict[str, dict] = {}
_boxscore_cache_times: dict[str, float] = {}
_BOXSCORE_TTL = 30  # always short — used during live games

_ESPN_TO_NBA: dict[str, str] = {
    "GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP",
}

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _normalize_abbr(abbr: str) -> str:
    a = abbr.strip().upper()
    return _ESPN_TO_NBA.get(a, a)


def get_scoreboard() -> list[dict[str, Any]]:
    """
    Returns today's NBA games.
    Each entry: {
        espn_game_id, home_abbr, away_abbr,
        status: "scheduled"|"live"|"final"|"postponed",
        status_detail: "7:30 PM ET" | "Q2 5:32" | "Final",
        start_time_utc: ISO string or None,
    }
    Cached 5 minutes.
    """
    global _scoreboard_cache, _scoreboard_cache_time
    now = time.time()
    any_live = any(g["status"] == "live" for g in _scoreboard_cache)
    ttl = _SCOREBOARD_TTL_LIVE if any_live else _SCOREBOARD_TTL_IDLE
    if _scoreboard_cache and (now - _scoreboard_cache_time) < ttl:
        return _scoreboard_cache

    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            timeout=10,
            headers=_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        games = []
        for event in data.get("events", []):
            comp = (event.get("competitions") or [{}])[0]
            status_type = comp.get("status", {}).get("type", {})
            status_name = status_type.get("name", "")

            if "IN_PROGRESS" in status_name:
                status = "live"
            elif status_name == "STATUS_FINAL":
                status = "final"
            elif "POSTPONE" in status_name or "CANCEL" in status_name:
                status = "postponed"
            else:
                status = "scheduled"

            home_abbr = away_abbr = ""
            home_score = away_score = None
            home_record = away_record = ""
            home_name = away_name = ""
            for competitor in comp.get("competitors", []):
                abbr = _normalize_abbr(competitor.get("team", {}).get("abbreviation", ""))
                name = competitor.get("team", {}).get("displayName", abbr)
                score = competitor.get("score")
                record = (competitor.get("records") or [{}])[0].get("summary", "")
                if competitor.get("homeAway") == "home":
                    home_abbr, home_name, home_score, home_record = abbr, name, score, record
                else:
                    away_abbr, away_name, away_score, away_record = abbr, name, score, record

            games.append({
                "espn_game_id": str(event.get("id", "")),
                "home_abbr": home_abbr,
                "home_name": home_name,
                "home_score": home_score,
                "home_record": home_record,
                "away_abbr": away_abbr,
                "away_name": away_name,
                "away_score": away_score,
                "away_record": away_record,
                "status": status,
                "status_detail": status_type.get("shortDetail", ""),
                "period_detail": status_type.get("detail", ""),
                "start_time_utc": comp.get("date"),
            })

        _scoreboard_cache = games
        _scoreboard_cache_time = now
        return games

    except Exception as exc:
        print(f"ESPN scoreboard fetch error: {exc}")
        return _scoreboard_cache


def get_game_box_score(espn_game_id: str) -> dict[str, Any]:
    """
    Returns full box score for a game.
    {
        game: {...scoreboard entry...},
        teams: [
            {
                abbr, name,
                players: [{name, minutes, points, rebounds, assists, steals, blocks, fg, three, ft, plus_minus, starter}]
            }
        ]
    }
    Cached 30s.
    """
    now = time.time()
    if espn_game_id in _boxscore_cache and (now - _boxscore_cache_times.get(espn_game_id, 0)) < _BOXSCORE_TTL:
        return _boxscore_cache[espn_game_id]

    game_info = next((g for g in get_scoreboard() if g["espn_game_id"] == espn_game_id), None)

    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
            params={"event": espn_game_id},
            timeout=10,
            headers=_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        teams = []
        for team_data in data.get("boxscore", {}).get("players", []):
            abbr = _normalize_abbr(team_data.get("team", {}).get("abbreviation", ""))
            name = team_data.get("team", {}).get("displayName", abbr)
            stats_block = (team_data.get("statistics") or [{}])[0]
            keys = stats_block.get("keys", [])

            def _idx(key):
                return keys.index(key) if key in keys else None

            i_min   = _idx("minutes")
            i_pts   = _idx("points")
            i_reb   = _idx("rebounds")
            i_ast   = _idx("assists")
            i_stl   = _idx("steals")
            i_blk   = _idx("blocks")
            i_fg    = _idx("fieldGoalsMade-fieldGoalsAttempted")
            i_three = _idx("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
            i_ft    = _idx("freeThrowsMade-freeThrowsAttempted")
            i_pm    = _idx("plusMinus")

            def _get(stats, i):
                return stats[i] if i is not None and i < len(stats) else "—"

            players = []
            for athlete in stats_block.get("athletes", []):
                stats = athlete.get("stats", [])
                if not stats:
                    continue
                players.append({
                    "name": athlete.get("athlete", {}).get("displayName", ""),
                    "starter": athlete.get("starter", False),
                    "active": athlete.get("active", True),
                    "minutes": _get(stats, i_min),
                    "points": _get(stats, i_pts),
                    "rebounds": _get(stats, i_reb),
                    "assists": _get(stats, i_ast),
                    "steals": _get(stats, i_stl),
                    "blocks": _get(stats, i_blk),
                    "fg": _get(stats, i_fg),
                    "three": _get(stats, i_three),
                    "ft": _get(stats, i_ft),
                    "plus_minus": _get(stats, i_pm),
                })

            # starters first, then bench, both sorted by minutes desc
            def _min_val(p):
                try:
                    return float(p["minutes"])
                except (ValueError, TypeError):
                    return 0.0

            starters = sorted([p for p in players if p["starter"]], key=_min_val, reverse=True)
            bench = sorted([p for p in players if not p["starter"]], key=_min_val, reverse=True)
            teams.append({"abbr": abbr, "name": name, "players": starters + bench})

        result = {"game": game_info, "teams": teams}
        _boxscore_cache[espn_game_id] = result
        _boxscore_cache_times[espn_game_id] = now
        return result

    except Exception as exc:
        print(f"ESPN box score error (game {espn_game_id}): {exc}")
        return _boxscore_cache.get(espn_game_id, {"game": game_info, "teams": []})


def get_game_for_team(team_abbr: str) -> dict[str, Any] | None:
    """Returns today's game entry for the given team abbr, or None."""
    abbr = team_abbr.strip().upper()
    for game in get_scoreboard():
        if game["home_abbr"] == abbr or game["away_abbr"] == abbr:
            return game
    return None


def get_confirmed_starters(espn_game_id: str) -> dict[str, bool]:
    """
    Returns {player_name_lower: True} for all confirmed starters.
    Available from ESPN game summary ~1 hour before tip-off.
    Empty dict if data not yet available.
    Cached 5 minutes.
    """
    now = time.time()
    if espn_game_id in _lineup_cache and (now - _lineup_cache_times.get(espn_game_id, 0)) < _LINEUP_TTL:
        return _lineup_cache[espn_game_id]

    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
            params={"event": espn_game_id},
            timeout=10,
            headers=_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        starters: dict[str, bool] = {}
        for roster in data.get("rosters", []):
            for athlete in roster.get("roster", []):
                if not athlete.get("starter"):
                    continue
                name = (athlete.get("athlete", {}).get("displayName") or "").lower().strip()
                if name:
                    starters[name] = True

        _lineup_cache[espn_game_id] = starters
        _lineup_cache_times[espn_game_id] = now
        return starters

    except Exception as exc:
        print(f"ESPN lineup fetch error (game {espn_game_id}): {exc}")
        return _lineup_cache.get(espn_game_id, {})


def get_player_game_status(player_name: str, team_abbr: str) -> dict[str, Any]:
    """
    Returns game and starter status for a player today.
    {
        game_status: "scheduled"|"live"|"final"|"postponed"|"no_game",
        status_detail: str,
        confirmed_starter: True | False | None  (None = lineup not yet released),
        is_home: bool | None,
    }
    """
    game = get_game_for_team(team_abbr)
    if not game:
        return {
            "game_status": "no_game",
            "status_detail": "No game today",
            "confirmed_starter": None,
            "is_home": None,
        }

    is_home = game["home_abbr"] == team_abbr.strip().upper()
    starters = get_confirmed_starters(game["espn_game_id"])

    confirmed_starter: bool | None = None
    if starters:
        name_lower = player_name.lower().strip()
        if name_lower in starters:
            confirmed_starter = True
        elif any(name_lower in s or s in name_lower for s in starters):
            confirmed_starter = True
        else:
            confirmed_starter = False  # lineup known, player not in starting five

    return {
        "game_status": game["status"],
        "status_detail": game["status_detail"],
        "confirmed_starter": confirmed_starter,
        "is_home": is_home,
    }
