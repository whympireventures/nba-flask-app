import threading
import time
from functools import lru_cache
from typing import Any
from datetime import datetime

import requests
from nba_api.stats.endpoints import commonplayerinfo, leaguedashplayerstats, playergamelog
from nba_api.stats.static import players as nba_static_players
from nba_api.library.http import NBAHTTP

from config import settings

# Bypass NBA.com bot detection by mimicking a real browser request
NBAHTTP.headers = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

_NBA_API_TIMEOUT = 60  # seconds — stats.nba.com can be slow
_GAME_LOG_TTL = 3600  # 1 hour — refresh game logs once per hour
_game_log_cache: dict[tuple, tuple[float, tuple]] = {}
_game_log_cache_lock = threading.Lock()


def _nba_api_fetch_with_retry(make_endpoint, max_attempts: int = 3, backoff: float = 3.0):
    """Call make_endpoint() and retry on connection/timeout errors."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return make_endpoint()
        except (OSError, requests.RequestException, requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


def _season_string(season_start_year: int) -> str:
    return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"


def _normalize_player_name(value: str) -> str:
    return " ".join(str(value).lower().replace(".", "").split())


def _normalize_game_date(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%b %d %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text


@lru_cache(maxsize=4)
def _season_player_dashboard(season_start_year: int) -> list[dict[str, Any]]:
    season = _season_string(season_start_year)
    def _fetch():
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=_NBA_API_TIMEOUT,
        )
        return endpoint.get_data_frames()[0].to_dict(orient="records")
    return _nba_api_fetch_with_retry(_fetch)


@lru_cache(maxsize=8)
def _recent_player_dashboard(season_start_year: int, last_n_games: int) -> list[dict[str, Any]]:
    season = _season_string(season_start_year)
    def _fetch():
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            last_n_games=str(last_n_games),
            timeout=_NBA_API_TIMEOUT,
        )
        return endpoint.get_data_frames()[0].to_dict(orient="records")
    return _nba_api_fetch_with_retry(_fetch)


def _cached_player_game_logs(player_id: str, season_start_year: int) -> tuple:
    """Cached per-player game log fetch — expires after 1 hour."""
    key = (player_id, season_start_year)
    now = time.time()
    with _game_log_cache_lock:
        if key in _game_log_cache:
            cached_at, result = _game_log_cache[key]
            if now - cached_at < _GAME_LOG_TTL:
                return result

    season = _season_string(season_start_year)
    team_lookup = {player["id"]: player for player in nba_static_players.get_players()}
    static_player = team_lookup.get(int(player_id)) if str(player_id).isdigit() else None

    def _fetch():
        endpoint = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=_NBA_API_TIMEOUT)
        frame = endpoint.get_data_frames()[0]
        if frame.empty:
            return ()
        results = []
        for row in frame.to_dict(orient="records"):
            matchup = str(row.get("MATCHUP", ""))
            opponent_abbr = matchup.split()[-1] if matchup else None
            results.append(
                {
                    "game": {
                        "id": row.get("Game_ID") or row.get("GAME_ID"),
                        "date": row.get("GAME_DATE"),
                    },
                    "team": {
                        "id": row.get("TEAM_ID"),
                        "code": row.get("TEAM_ABBREVIATION"),
                    },
                    "player": {
                        "id": player_id,
                        "firstname": (static_player or {}).get("first_name"),
                        "lastname": (static_player or {}).get("last_name"),
                    },
                    "opponent_abbr": opponent_abbr,
                    "points": row.get("PTS", 0),
                    "fgm": row.get("FGM", 0),
                    "fga": row.get("FGA", 0),
                    "fgp": row.get("FG_PCT", 0),
                    "ftp": row.get("FT_PCT", 0),
                    "tpm": row.get("FG3M", 0),
                    "tpa": row.get("FG3A", 0),
                    "tpp": row.get("FG3_PCT", 0),
                    "offReb": row.get("OREB", 0),
                    "defReb": row.get("DREB", 0),
                    "totReb": row.get("REB", 0),
                    "assists": row.get("AST", 0),
                    "pFouls": row.get("PF", 0),
                    "steals": row.get("STL", 0),
                    "turnovers": row.get("TOV", 0),
                    "blocks": row.get("BLK", 0),
                    "plusMinus": row.get("PLUS_MINUS", 0),
                    "min": row.get("MIN", 0),
                }
            )
        return tuple(results)

    result = _nba_api_fetch_with_retry(_fetch)
    with _game_log_cache_lock:
        _game_log_cache[key] = (now, result)
    return result


class NBAApiClient:
    def __init__(self) -> None:
        self.base_url = f"https://{settings.rapidapi_host}"
        self.session = requests.Session()
        if settings.rapidapi_key:
            self.session.headers.update(
                {
                    "X-RapidAPI-Key": settings.rapidapi_key,
                    "X-RapidAPI-Host": settings.rapidapi_host,
                }
            )

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not settings.rapidapi_key:
            raise RuntimeError("RAPIDAPI_KEY is not configured.")

        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _split_height(height: str | None) -> tuple[str | None, str | None]:
        if not height or "-" not in str(height):
            return None, None
        feet, inches = str(height).split("-", 1)
        return feet, inches

    def _format_static_player(self, player: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": player["id"],
            "firstname": player["first_name"],
            "lastname": player["last_name"],
            "birth": {"date": None, "country": None},
            "nba": {"start": player.get("from_year"), "pro": None},
            "height": {"feets": None, "inches": None},
            "weight": {"pounds": None},
            "college": None,
            "leagues": {"standard": {"jersey": None, "pos": None}},
        }

    def _format_common_player_info(self, row: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        feet, inches = self._split_height(row.get("HEIGHT"))
        return {
            "id": row.get("PERSON_ID") or (fallback or {}).get("id"),
            "firstname": row.get("FIRST_NAME") or (fallback or {}).get("firstname"),
            "lastname": row.get("LAST_NAME") or (fallback or {}).get("lastname"),
            "birth": {
                "date": row.get("BIRTHDATE"),
                "country": row.get("COUNTRY"),
            },
            "nba": {
                "start": row.get("FROM_YEAR") or ((fallback or {}).get("nba") or {}).get("start"),
                "pro": row.get("SEASON_EXP"),
            },
            "height": {
                "feets": feet,
                "inches": inches,
            },
            "weight": {"pounds": row.get("WEIGHT")},
            "college": row.get("SCHOOL"),
            "leagues": {
                "standard": {
                    "jersey": row.get("JERSEY"),
                    "pos": row.get("POSITION"),
                }
            },
        }

    def _search_players_nba_api(self, query: str) -> list[dict[str, Any]]:
        query_lower = query.lower().strip()
        found_players = nba_static_players.find_players_by_full_name(query)
        if not found_players:
            found_players = [
                player
                for player in nba_static_players.get_players()
                if query_lower in player["last_name"].lower()
                or query_lower in player["full_name"].lower()
            ]
        found_players = sorted(
            found_players,
            key=lambda player: (
                player["full_name"].lower() != query_lower,
                player["last_name"].lower() != query_lower,
                not player.get("is_active", False),
                player["full_name"],
            ),
        )
        return [self._format_static_player(player) for player in found_players[:20]]

    def _get_player_details_nba_api(self, player_id: str) -> dict[str, Any] | None:
        fallback_candidates = [player for player in nba_static_players.get_players() if str(player["id"]) == str(player_id)]
        fallback = self._format_static_player(fallback_candidates[0]) if fallback_candidates else None
        try:
            endpoint = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=_NBA_API_TIMEOUT)
            frame = endpoint.get_data_frames()[0]
            if frame.empty:
                return fallback
            return self._format_common_player_info(frame.iloc[0].to_dict(), fallback=fallback)
        except Exception:
            return fallback

    def _get_player_statistics_nba_api(self, player_id: str, season_start_year: int | None = None) -> list[dict[str, Any]]:
        season_year = season_start_year if season_start_year is not None else settings.season_start_year
        return list(_cached_player_game_logs(str(player_id), int(season_year)))


    def search_players(self, query: str) -> list[dict[str, Any]]:
        if settings.rapidapi_key:
            data = self._get("/players", {"search": query})
            return data.get("response", [])
        return self._search_players_nba_api(query)

    def get_player_details(self, player_id: str) -> dict[str, Any] | None:
        if settings.rapidapi_key:
            data = self._get("/players", {"id": str(player_id)})
            players = data.get("response", [])
            return players[0] if players else None
        return self._get_player_details_nba_api(player_id)

    def get_player_statistics(self, player_id: str, season_start_year: int | None = None) -> list[dict[str, Any]]:
        if settings.rapidapi_key:
            season = season_start_year if season_start_year is not None else settings.season_start_year
            data = self._get("/players/statistics", {"id": str(player_id), "season": str(season)})
            return data.get("response", [])
        return self._get_player_statistics_nba_api(player_id, season_start_year=season_start_year)

    def get_player_actual_result(
        self,
        player_id: str,
        *,
        game_date: str,
        opponent_abbr: str | None = None,
    ) -> dict[str, float] | None:
        target_date = _normalize_game_date(game_date)
        if not target_date:
            return None

        season_year = int(target_date[:4])
        candidate_seasons = sorted({season_year, season_year - 1}, reverse=True)
        normalized_opponent = str(opponent_abbr or "").strip().upper()

        for season_start_year in candidate_seasons:
            try:
                game_logs = self.get_player_statistics(player_id, season_start_year=season_start_year)
            except Exception:
                continue
            matches = []
            for row in game_logs:
                raw_date = (
                    ((row.get("game") or {}).get("date"))
                    if isinstance(row.get("game"), dict)
                    else row.get("GAME_DATE")
                )
                row_date = _normalize_game_date(raw_date)
                if row_date != target_date:
                    continue
                row_opponent = str(row.get("opponent_abbr") or "").upper()
                if normalized_opponent and row_opponent and row_opponent != normalized_opponent:
                    continue
                matches.append(row)
            if not matches:
                continue

            selected = matches[0]
            return {
                "actual_points": float(selected.get("points", 0)),
                "actual_assists": float(selected.get("assists", 0)),
                "actual_rebounds": float(selected.get("totReb", 0)),
            }
        return None

    def resolve_player_id_by_name(self, full_name: str) -> int | None:
        normalized_target = _normalize_player_name(full_name)
        exact_matches = [
            player
            for player in nba_static_players.get_players()
            if _normalize_player_name(player.get("full_name", "")) == normalized_target
        ]
        if exact_matches:
            exact_matches.sort(
                key=lambda player: (
                    not player.get("is_active", False),
                    player.get("from_year") is None,
                    -(player.get("from_year") or 0),
                )
            )
            return int(exact_matches[0]["id"])

        partial_matches = [
            player
            for player in nba_static_players.get_players()
            if normalized_target in _normalize_player_name(player.get("full_name", ""))
        ]
        if not partial_matches:
            return None
        partial_matches.sort(
            key=lambda player: (
                not player.get("is_active", False),
                player.get("from_year") is None,
                -(player.get("from_year") or 0),
            )
        )
        return int(partial_matches[0]["id"])

    def get_team_rotation(self, team_abbr: str, limit: int = 5) -> list[dict[str, Any]]:
        season_year = settings.season_start_year
        season_rotation = [
            row
            for row in _season_player_dashboard(season_year)
            if str(row.get("TEAM_ABBREVIATION", "")).upper() == team_abbr.upper()
        ]
        recent_rotation = {
            int(row["PLAYER_ID"]): row
            for row in _recent_player_dashboard(season_year, last_n_games=10)
            if row.get("PLAYER_ID") is not None
            and str(row.get("TEAM_ABBREVIATION", "")).upper() == team_abbr.upper()
        }

        rotation = [
            {
                "id": int(row["PLAYER_ID"]),
                "name": str(row.get("PLAYER_NAME", "")),
                "team_abbr": str(row.get("TEAM_ABBREVIATION", "")).upper(),
                "minutes": float(row.get("MIN", 0.0)),
                "recent_minutes": float(recent_rotation.get(int(row["PLAYER_ID"]), {}).get("MIN", row.get("MIN", 0.0))),
                "games_played": int(row.get("GP", 0)),
                "usage_rate": float(row.get("USG_PCT", 0.0)),
                "recent_usage_rate": float(recent_rotation.get(int(row["PLAYER_ID"]), {}).get("USG_PCT", row.get("USG_PCT", 0.0))),
            }
            for row in season_rotation
            if row.get("PLAYER_ID") is not None and int(row.get("GP", 0)) > 0
        ]
        for player in rotation:
            player["projected_minutes"] = (player["recent_minutes"] * 0.6) + (player["minutes"] * 0.4)
            player["minutes_trend"] = player["recent_minutes"] - player["minutes"]
            player["role_label"] = "Core"
            if player["projected_minutes"] < 22:
                player["role_label"] = "Bench Risk"
            elif player["projected_minutes"] < 28:
                player["role_label"] = "Rotation"
            elif player["projected_minutes"] >= 34:
                player["role_label"] = "Heavy Load"

        rotation.sort(
            key=lambda player: (
                player["projected_minutes"],
                player["recent_minutes"],
                player["games_played"],
                player["recent_usage_rate"],
                player["usage_rate"],
            ),
            reverse=True,
        )
        return rotation[:limit]
