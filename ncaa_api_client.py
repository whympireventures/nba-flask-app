from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Any

import requests

from config import settings


_ESPN_SITE_API_BASE = "https://site.api.espn.com/apis/site/v2"
_ESPN_WEB_API_BASE = "https://site.web.api.espn.com/apis/common/v3"
_SPORT_PATH = "/sports/basketball/mens-college-basketball"
_DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
}


def default_ncaa_season_year() -> int:
    today = datetime.now()
    return today.year if today.month >= 10 else today.year - 1


def _normalize_name(value: str) -> str:
    return " ".join(str(value or "").lower().replace(".", "").split())


def _normalize_stat_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _safe_float(value: Any) -> float:
    if value in (None, "", "--", "null", "None"):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        try:
            return float(minutes) + (float(seconds) / 60.0)
        except ValueError:
            return 0.0
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def _extract_numeric_id(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    if text.isdigit():
        return text
    match = re.search(r"(?:athletes?|teams?|a:|t:|id/)(\d+)", text)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", text)
    return digits[-1] if digits else ""


def _normalize_percentage(value: Any) -> float:
    number = _safe_float(value)
    if 0.0 < number <= 1.0:
        return number * 100.0
    return number


def _parse_made_attempts(value: Any) -> tuple[float, float]:
    text = str(value or "").strip()
    if "-" not in text:
        return 0.0, 0.0
    made, attempts = text.split("-", 1)
    return _safe_float(made), _safe_float(attempts)


def _iter_dict_nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_dict_nodes(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dict_nodes(item)


def _pick_stat_value(stat_map: dict[str, Any], *aliases: str) -> Any:
    normalized_map = {
        _normalize_stat_label(key): value
        for key, value in stat_map.items()
    }
    for alias in aliases:
        normalized_alias = _normalize_stat_label(alias)
        if normalized_alias in normalized_map:
            return normalized_map[normalized_alias]
    return None


def _extract_event_metadata(entry: dict[str, Any], team_id: str | None) -> tuple[str, str, str, str]:
    event = entry.get("event") if isinstance(entry.get("event"), dict) else {}
    competition = event.get("competition") if isinstance(event.get("competition"), dict) else {}
    if not competition and isinstance(entry.get("competition"), dict):
        competition = entry["competition"]

    raw_game_id = (
        competition.get("id")
        or event.get("id")
        or entry.get("id")
        or competition.get("$ref")
        or event.get("$ref")
    )
    raw_date = competition.get("date") or event.get("date") or entry.get("date")

    opponent_abbr = str(entry.get("opponent_abbr") or "").upper()
    opponent_name = str(entry.get("opponent_name") or "").strip()
    competitors = competition.get("competitors") if isinstance(competition.get("competitors"), list) else []
    if competitors:
        selected = None
        for competitor in competitors:
            team = competitor.get("team") if isinstance(competitor, dict) else {}
            competitor_id = _extract_numeric_id(team.get("id") or competitor.get("id"))
            if team_id and competitor_id and competitor_id == str(team_id):
                continue
            selected = competitor
            if team_id:
                break
        if selected is None:
            selected = competitors[-1]
        selected_team = selected.get("team") if isinstance(selected, dict) else {}
        opponent_abbr = str(
            selected_team.get("abbreviation")
            or selected_team.get("shortDisplayName")
            or opponent_abbr
        ).upper()
        opponent_name = str(
            selected_team.get("displayName")
            or selected_team.get("name")
            or opponent_name
        ).strip()

    return (
        _extract_numeric_id(raw_game_id),
        str(raw_date or ""),
        opponent_abbr,
        opponent_name,
    )


def _normalize_gamelog_entry(
    stat_map: dict[str, Any],
    entry: dict[str, Any],
    *,
    player_id: str,
    team_id: str | None,
) -> dict[str, Any] | None:
    points = _safe_float(_pick_stat_value(stat_map, "PTS", "POINTS"))
    rebounds = _safe_float(_pick_stat_value(stat_map, "REB", "TREB", "REBOUNDS"))
    assists = _safe_float(_pick_stat_value(stat_map, "AST", "ASSISTS"))
    minutes = _safe_float(_pick_stat_value(stat_map, "MIN", "MINUTES"))
    turnovers = _safe_float(_pick_stat_value(stat_map, "TO", "TOV", "TURNOVERS"))
    fg_pct = _normalize_percentage(_pick_stat_value(stat_map, "FG%", "FGPCT", "FG_PCT"))
    three_pt_pct = _normalize_percentage(_pick_stat_value(stat_map, "3PT%", "3P%", "FG3%", "3PTPCT", "3PPCT"))

    fgm, fga = _parse_made_attempts(_pick_stat_value(stat_map, "FG", "FGM-A"))
    tpm, tpa = _parse_made_attempts(_pick_stat_value(stat_map, "3PT", "3PTM-A", "3PM-A", "FG3"))

    if minutes <= 0 and not any((points, rebounds, assists, turnovers, fg_pct, three_pt_pct)):
        return None

    game_id, raw_date, opponent_abbr, opponent_name = _extract_event_metadata(entry, team_id)
    return {
        "game": {
            "id": game_id,
            "date": raw_date,
        },
        "player": {
            "id": str(player_id),
        },
        "opponent_abbr": opponent_abbr,
        "opponent_name": opponent_name,
        "points": points,
        "totReb": rebounds,
        "assists": assists,
        "turnovers": turnovers,
        "fgm": fgm,
        "fga": fga,
        "fgp": fg_pct,
        "tpm": tpm,
        "tpa": tpa,
        "tpp": three_pt_pct,
        "min": minutes,
    }


def _extract_gamelog_rows_from_section(
    section: dict[str, Any],
    *,
    player_id: str,
    team_id: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories = section.get("categories") if isinstance(section.get("categories"), list) else []

    if categories:
        for category in categories:
            labels = category.get("labels") if isinstance(category.get("labels"), list) else []
            entries = category.get("events") if isinstance(category.get("events"), list) else []
            if not entries:
                entries = category.get("items") if isinstance(category.get("items"), list) else []
            for entry in entries:
                stats = entry.get("stats") if isinstance(entry.get("stats"), list) else entry.get("statistics")
                if isinstance(stats, list):
                    stat_map = {str(label): stats[index] for index, label in enumerate(labels) if index < len(stats)}
                elif isinstance(stats, dict):
                    stat_map = stats
                else:
                    stat_map = {}
                normalized = _normalize_gamelog_entry(
                    stat_map,
                    entry,
                    player_id=player_id,
                    team_id=team_id,
                )
                if normalized:
                    rows.append(normalized)
        return rows

    labels = section.get("labels") if isinstance(section.get("labels"), list) else []
    entries = section.get("events") if isinstance(section.get("events"), list) else []
    if not entries:
        entries = section.get("items") if isinstance(section.get("items"), list) else []
    for entry in entries:
        stats = entry.get("stats") if isinstance(entry.get("stats"), list) else entry.get("statistics")
        if isinstance(stats, list):
            stat_map = {str(label): stats[index] for index, label in enumerate(labels) if index < len(stats)}
        elif isinstance(stats, dict):
            stat_map = stats
        else:
            stat_map = {}
        normalized = _normalize_gamelog_entry(
            stat_map,
            entry,
            player_id=player_id,
            team_id=team_id,
        )
        if normalized:
            rows.append(normalized)
    return rows


def _extract_gamelog_rows(payload: dict[str, Any], *, player_id: str, team_id: str | None) -> list[dict[str, Any]]:
    # ESPN v3 structure: top-level labels + top-level events dict + seasonTypes[].categories[].events[{eventId,stats}]
    top_labels = payload.get("labels") if isinstance(payload.get("labels"), list) else []
    events_lookup: dict[str, dict[str, Any]] = {}
    raw_events = payload.get("events")
    if isinstance(raw_events, dict):
        events_lookup = raw_events

    rows: list[dict[str, Any]] = []

    if top_labels and events_lookup:
        season_types = payload.get("seasonTypes") if isinstance(payload.get("seasonTypes"), list) else []
        for season_type in season_types:
            categories = season_type.get("categories") if isinstance(season_type.get("categories"), list) else []
            for category in categories:
                entries = category.get("events") if isinstance(category.get("events"), list) else []
                for entry in entries:
                    event_id = str(entry.get("eventId") or "")
                    stats = entry.get("stats") if isinstance(entry.get("stats"), list) else []
                    if not stats:
                        continue
                    stat_map = {str(top_labels[i]): stats[i] for i in range(min(len(top_labels), len(stats)))}
                    event_obj = events_lookup.get(event_id, {})
                    opponent = event_obj.get("opponent") if isinstance(event_obj.get("opponent"), dict) else {}
                    opponent_abbr = str(opponent.get("abbreviation") or "").upper()
                    game_date = str(event_obj.get("gameDate") or "")
                    enriched_entry = {"id": event_id, "date": game_date, "opponent_abbr": opponent_abbr}
                    normalized = _normalize_gamelog_entry(stat_map, enriched_entry, player_id=player_id, team_id=team_id)
                    if normalized:
                        rows.append(normalized)
        if rows:
            return rows

    # Fallback: legacy section-based parsing
    season_types = payload.get("seasonTypes") if isinstance(payload.get("seasonTypes"), list) else []
    for season_type in season_types:
        rows.extend(_extract_gamelog_rows_from_section(season_type, player_id=player_id, team_id=team_id))

    if not rows:
        for node in _iter_dict_nodes(payload):
            candidate_rows = _extract_gamelog_rows_from_section(node, player_id=player_id, team_id=team_id)
            if candidate_rows:
                rows.extend(candidate_rows)

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        game = row.get("game", {})
        key = (str(game.get("id") or ""), str(game.get("date") or ""))
        deduped[key] = row
    return list(deduped.values())


def _athlete_candidate_from_node(node: dict[str, Any], query: str) -> dict[str, Any] | None:
    athlete = node.get("athlete") if isinstance(node.get("athlete"), dict) else node
    athlete_id = _extract_numeric_id(
        athlete.get("id")
        or athlete.get("uid")
        or athlete.get("$ref")
        or athlete.get("href")
        or node.get("href")
    )
    display_name = str(
        athlete.get("displayName")
        or athlete.get("fullName")
        or athlete.get("shortName")
        or athlete.get("name")
        or node.get("displayName")
        or node.get("name")
        or ""
    ).strip()
    if not athlete_id or not display_name:
        return None

    normalized_name = _normalize_name(display_name)
    normalized_query = _normalize_name(query)
    if normalized_query and normalized_query not in normalized_name:
        return None

    team = athlete.get("team") if isinstance(athlete.get("team"), dict) else {}
    if not team and isinstance(node.get("team"), dict):
        team = node["team"]
    team_id = _extract_numeric_id(team.get("id") or team.get("uid") or team.get("$ref"))

    return {
        "id": athlete_id,
        "display_name": display_name,
        "team_id": team_id,
        "team_name": str(team.get("displayName") or team.get("name") or "").strip(),
        "team_abbr": str(team.get("abbreviation") or team.get("shortDisplayName") or "").upper(),
    }


@lru_cache(maxsize=256)
def _search_players_cached(query: str) -> tuple[dict[str, Any], ...]:
    if not query.strip():
        return ()

    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)
    endpoints = [
        (
            f"{_ESPN_WEB_API_BASE}/search",
            {"query": query, "limit": 20, "type": "player"},
        ),
        (
            f"{_ESPN_SITE_API_BASE}{_SPORT_PATH}/athletes",
            {"search": query, "limit": 20},
        ),
    ]

    for url, params in endpoints:
        try:
            response = session.get(url, params=params, timeout=settings.request_timeout)
            if response.status_code >= 400:
                continue
            payload = response.json()
        except Exception:
            continue

        candidates: dict[str, dict[str, Any]] = {}
        for node in _iter_dict_nodes(payload):
            candidate = _athlete_candidate_from_node(node, query)
            if not candidate:
                continue
            existing = candidates.get(candidate["id"])
            if existing is None or (not existing.get("team_id") and candidate.get("team_id")):
                candidates[candidate["id"]] = candidate

        if candidates:
            ordered = sorted(
                candidates.values(),
                key=lambda player: (
                    _normalize_name(player["display_name"]) != _normalize_name(query),
                    player["display_name"],
                ),
            )
            return tuple(ordered[:20])

    return ()


@lru_cache(maxsize=512)
def _player_gamelogs_cached(player_id: str, season: int, team_id: str | None) -> tuple[dict[str, Any], ...]:
    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)
    endpoint_candidates = [
        (
            f"{_ESPN_WEB_API_BASE}{_SPORT_PATH}/athletes/{player_id}/gamelog",
            {"season": season},
        ),
        (
            f"{_ESPN_WEB_API_BASE}{_SPORT_PATH}/athletes/{player_id}/statistics/0",
            {"season": season},
        ),
    ]
    if team_id:
        endpoint_candidates.insert(
            0,
            (
                f"{_ESPN_WEB_API_BASE}{_SPORT_PATH}/teams/{team_id}/athletes/{player_id}/gamelog",
                {"season": season},
            ),
        )

    for url, params in endpoint_candidates:
        try:
            response = session.get(url, params=params, timeout=settings.request_timeout)
            if response.status_code >= 400:
                continue
            payload = response.json()
        except Exception:
            continue

        rows = _extract_gamelog_rows(payload, player_id=player_id, team_id=team_id)
        if rows:
            return tuple(rows)

    return ()


class NCAAApiClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(_DEFAULT_HEADERS)

    def search_players(self, query: str) -> list[dict[str, Any]]:
        return list(_search_players_cached(query))

    def resolve_player_by_name(self, full_name: str) -> dict[str, Any] | None:
        players = self.search_players(full_name)
        if not players:
            return None
        normalized_target = _normalize_name(full_name)
        exact_matches = [
            player for player in players if _normalize_name(player["display_name"]) == normalized_target
        ]
        if exact_matches:
            return exact_matches[0]
        return players[0]

    def resolve_player_id_by_name(self, full_name: str) -> str | None:
        player = self.resolve_player_by_name(full_name)
        if player is None:
            return None
        return str(player["id"])

    def get_player_statistics(
        self,
        player_id: str,
        *,
        season: int | None = None,
        team_id: str | None = None,
    ) -> list[dict[str, Any]]:
        season_value = int(season if season is not None else default_ncaa_season_year())
        return list(_player_gamelogs_cached(str(player_id), season_value, str(team_id or "") or None))
