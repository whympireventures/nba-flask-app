"""
ESPN free injury report client.
Endpoint: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
No API key required. Cached for 1 hour per server session.
"""
from __future__ import annotations

import threading
import time
import requests

_CACHE_TTL = 3600  # seconds

# ESPN uses different abbreviations than nba_api in a few cases
_ESPN_TO_NBA: dict[str, str] = {
    "GS":  "GSW",
    "NY":  "NYK",
    "SA":  "SAS",
    "NO":  "NOP",
    "OKC": "OKC",
    "PHX": "PHX",
}

_OUT_STATUSES = {"out", "doubtful"}
_QUESTIONABLE_STATUSES = {"questionable", "probable", "day-to-day"}

_cache: dict[str, list[dict]] = {}
_cache_time: float = 0.0
_cache_lock = threading.Lock()


def _normalize_abbr(espn_abbr: str) -> str:
    abbr = espn_abbr.strip().upper()
    return _ESPN_TO_NBA.get(abbr, abbr)


def fetch_injury_report() -> dict[str, list[dict]]:
    """
    Returns {team_abbr: [{"player_name", "status", "description"}]}
    Cached for 1 hour. Returns last known data on fetch error.
    """
    global _cache, _cache_time
    now = time.time()
    with _cache_lock:
        if _cache and (now - _cache_time) < _CACHE_TTL:
            return _cache

    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        result: dict[str, list[dict]] = {}
        for team_entry in data.get("injuries", []):
            team = team_entry.get("team", {})
            raw_abbr = team.get("abbreviation", "")
            if not raw_abbr:
                continue
            team_abbr = _normalize_abbr(raw_abbr)
            players = []
            for inj in team_entry.get("injuries", []):
                athlete = inj.get("athlete", {})
                name = athlete.get("displayName", "").strip()
                status = inj.get("status", "").strip()
                description = inj.get("longComment", "") or inj.get("shortComment", "") or ""
                if name:
                    players.append({
                        "player_name": name,
                        "status": status,
                        "status_key": status.lower().replace("-", " "),
                        "description": description.strip(),
                        "is_out": status.lower().replace("-", " ") in _OUT_STATUSES,
                        "is_questionable": status.lower().replace("-", " ") in _QUESTIONABLE_STATUSES,
                    })
            result[team_abbr] = players

        with _cache_lock:
            _cache = result
            _cache_time = now
        return result

    except Exception as exc:
        with _cache_lock:
            return _cache


def get_team_injuries(team_abbr: str) -> list[dict]:
    """All injury entries for a team."""
    return fetch_injury_report().get(team_abbr.upper(), [])


def get_out_player_names(team_abbr: str) -> set[str]:
    """Lowercase names of players confirmed Out or Doubtful."""
    return {
        inj["player_name"].lower()
        for inj in get_team_injuries(team_abbr)
        if inj["is_out"]
    }


def get_player_status(player_name: str, team_abbr: str) -> dict | None:
    """
    Returns the injury dict for a player, or None if healthy/not listed.
    Matches by full name or partial last name.
    """
    name_lower = player_name.lower().strip()
    for inj in get_team_injuries(team_abbr):
        inj_name = inj["player_name"].lower()
        if inj_name == name_lower or name_lower in inj_name or inj_name in name_lower:
            return inj
    return None


def get_out_minute_total(team_abbr: str, rotation: list[dict]) -> float:
    """
    Given a rotation (list of player dicts with 'name' and 'minutes'),
    return total season-avg minutes belonging to confirmed Out/Doubtful players.
    """
    out_names = get_out_player_names(team_abbr)
    total = 0.0
    for player in rotation:
        if player.get("name", "").lower() in out_names:
            total += float(player.get("minutes", 0.0))
    return total
