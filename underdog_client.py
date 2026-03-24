from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Iterable

import requests

from config import settings


MARKET_KEY_BY_LABEL = {
    "points": "line_points",
    "assists": "line_assists",
    "rebounds": "line_rebounds",
    "pts + rebs": "line_points_rebounds",
    "pts + rebounds": "line_points_rebounds",
    "points + rebounds": "line_points_rebounds",
    "pts + asts": "line_points_assists",
    "pts + assists": "line_points_assists",
    "points + assists": "line_points_assists",
    "rebounds + assists": "line_assists_rebounds",
    "assists + rebounds": "line_assists_rebounds",
    "pts + rebs + asts": "line_points_rebounds_assists",
    "pts + rebounds + assists": "line_points_rebounds_assists",
    "points + rebounds + assists": "line_points_rebounds_assists",
    "pra": "line_points_rebounds_assists",
}
MARKET_LABEL_BY_KEY = {
    "line_points": "Points",
    "line_assists": "Assists",
    "line_rebounds": "Rebounds",
    "line_points_rebounds": "Points + Rebounds",
    "line_points_assists": "Points + Assists",
    "line_assists_rebounds": "Assists + Rebounds",
    "line_points_rebounds_assists": "PRA",
}
SUPPORTED_FULL_SLATE_MARKETS = {
    "Points",
    "Assists",
    "Rebounds",
    "Points + Rebounds",
    "Points + Assists",
    "Rebounds + Assists",
    "Assists + Rebounds",
    "Pts + Rebs + Asts",
    "PRA",
}


def _normalize_name(value: str) -> str:
    return " ".join(str(value).lower().replace(".", "").split())


def _normalize_market_label(value: str | None) -> str:
    normalized = " ".join(str(value or "").strip().lower().replace("&", "+").split())
    normalized = normalized.replace("rebs", "rebounds").replace("asts", "assists")
    return normalized


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        for record in value.values():
            if isinstance(record, dict):
                yield record
    elif isinstance(value, list):
        for record in value:
            if isinstance(record, dict):
                yield record


@dataclass(frozen=True)
class UnderdogBoardEntry:
    player_name: str
    market_key: str
    market_label: str
    line_score: float
    opponent_abbr: str
    start_time: str | None
    selection_key: str | None = None
    selection_label: str | None = None
    payout_multiplier: float | None = None
    line_type: str | None = None
    sport: str = "nba"


class UnderdogProviderError(RuntimeError):
    pass


class UnderdogClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": settings.underdog_user_agent,
            }
        )

    def is_configured(self) -> bool:
        return all(
            (
                settings.underdog_api_base,
                settings.underdog_search_path,
                settings.underdog_lines_path,
                settings.underdog_product,
                settings.underdog_product_experience_id,
                settings.underdog_sport_id,
                settings.underdog_state_config_id,
            )
        )

    def configuration_message(self) -> str:
        if self.is_configured():
            return ""
        return "Set the Underdog API settings to load current Pick'em lines."

    def setup_hint(self) -> str:
        return (
            "Recommended env vars: UNDERDOG_API_BASE, UNDERDOG_SEARCH_PATH, UNDERDOG_LINES_PATH, UNDERDOG_PRODUCT, "
            "UNDERDOG_PRODUCT_EXPERIENCE_ID, UNDERDOG_SPORT_ID, and UNDERDOG_STATE_CONFIG_ID."
        )

    def _request(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        if not self.is_configured():
            raise UnderdogProviderError(self.configuration_message())
        last_exc = None
        for attempt in range(3):
            response = self.session.get(
                f"{settings.underdog_api_base.rstrip('/')}{path}",
                params=params,
                timeout=settings.request_timeout,
            )
            if response.status_code == 429:
                last_exc = requests.HTTPError(response=response)
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code in (401, 403):
                raise UnderdogProviderError("Underdog blocked the board request.")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise UnderdogProviderError("Underdog returned an unexpected response shape.")
            return payload
        raise UnderdogProviderError("Underdog rate limited after 3 attempts") from last_exc

    def _get_market_catalog(self, sport_id: str) -> dict[str, str]:
        payload = self._request(
            settings.underdog_search_path,
            {
                "product": settings.underdog_product,
                "product_experience_id": settings.underdog_product_experience_id,
                "sport_id": sport_id,
                "state_config_id": settings.underdog_state_config_id,
            },
        )
        market_catalog: dict[str, str] = {}
        for row in payload.get("over_under_lines", []):
            if not isinstance(row, dict):
                continue
            over_under = row.get("over_under") or {}
            appearance_stat = over_under.get("appearance_stat") or {}
            display_stat = str(
                appearance_stat.get("display_stat")
                or over_under.get("grid_display_title")
                or ""
            ).strip()
            pickem_stat_id = str(appearance_stat.get("pickem_stat_id") or "").strip()
            if display_stat and pickem_stat_id and display_stat in SUPPORTED_FULL_SLATE_MARKETS:
                market_catalog[display_stat] = pickem_stat_id
        return market_catalog

    def _get_full_market_lines(self, filter_id: str, sport_id: str) -> dict[str, Any]:
        return self._request(
            settings.underdog_lines_path,
            {
                "filter_id": filter_id,
                "filter_type": "PickemStat",
                "include_live": str(settings.underdog_include_live).lower(),
                "product": settings.underdog_product,
                "product_experience_id": settings.underdog_product_experience_id,
                "show_mass_option_markets": str(settings.underdog_show_mass_option_markets).lower(),
                "sport_id": sport_id,
                "state_config_id": settings.underdog_state_config_id,
            },
        )

    def _extract_market_key(self, row: dict[str, Any]) -> str | None:
        over_under = row.get("over_under") or {}
        appearance_stat = over_under.get("appearance_stat") or {}
        for candidate in (
            appearance_stat.get("display_stat"),
            over_under.get("grid_display_title"),
            over_under.get("title"),
        ):
            market_key = MARKET_KEY_BY_LABEL.get(_normalize_market_label(candidate))
            if market_key:
                return market_key
        return None

    def _extract_player_name(self, player_row: dict[str, Any], fallback_name: str = "") -> str:
        if player_row.get("name"):
            return str(player_row["name"]).strip()
        first_name = str(player_row.get("first_name") or "").strip()
        last_name = str(player_row.get("last_name") or "").strip()
        full_name = f"{first_name} {last_name}".strip()
        return full_name or fallback_name

    def _extract_opponent_abbr(
        self,
        appearance_row: dict[str, Any],
        game_row: dict[str, Any],
        teams_by_id: dict[str, dict[str, Any]],
    ) -> str:
        player_team_id = appearance_row.get("team_id")
        home_team_id = game_row.get("home_team_id")
        away_team_id = game_row.get("away_team_id")
        if player_team_id == home_team_id:
            return str((teams_by_id.get(str(away_team_id)) or {}).get("abbr") or "").upper()
        if player_team_id == away_team_id:
            return str((teams_by_id.get(str(home_team_id)) or {}).get("abbr") or "").upper()
        return ""

    def _fetch_board(self, sport: str = "nba") -> list[UnderdogBoardEntry]:
        sport_id = settings.underdog_ncaab_sport_id if sport == "ncaab" else settings.underdog_sport_id
        market_catalog = self._get_market_catalog(sport_id)
        entries: list[UnderdogBoardEntry] = []

        for _, filter_id in sorted(market_catalog.items()):
            payload = self._get_full_market_lines(filter_id, sport_id)
            appearances_by_id = {
                str(row.get("id")): row
                for row in _iter_records(payload.get("appearances"))
                if row.get("id")
            }
            games_by_id = {
                int(row.get("id")): row
                for row in _iter_records(payload.get("games"))
                if row.get("id") is not None
            }
            players_by_id = {
                str(row.get("id")): row
                for row in _iter_records(payload.get("players"))
                if row.get("id")
            }
            teams_by_id = {
                str(row.get("id")): row
                for row in _iter_records(payload.get("teams"))
                if row.get("id")
            }

            for row in _iter_records(payload.get("over_under_lines")):
                market_key = self._extract_market_key(row)
                if not market_key:
                    continue
                line_score = _parse_float(row.get("stat_value"))
                if line_score is None:
                    continue

                over_under = row.get("over_under") or {}
                appearance_stat = over_under.get("appearance_stat") or {}
                appearance_row = appearances_by_id.get(str(appearance_stat.get("appearance_id"))) or {}
                player_row = players_by_id.get(str(appearance_row.get("player_id"))) or {}
                player_name = self._extract_player_name(
                    player_row,
                    fallback_name=str(((row.get("options") or [{}])[0] or {}).get("selection_header") or "").strip(),
                )
                if not player_name:
                    continue
                game_row = games_by_id.get(int(appearance_row.get("match_id") or 0)) or {}
                start_time = str(game_row.get("scheduled_at") or "") or None
                opponent_abbr = self._extract_opponent_abbr(appearance_row, game_row, teams_by_id)

                for option in row.get("options", []):
                    if not isinstance(option, dict):
                        continue
                    choice = str(option.get("choice") or "").strip().lower()
                    if choice == "higher":
                        selection_key = "more"
                        selection_label = "Higher"
                    elif choice == "lower":
                        selection_key = "less"
                        selection_label = "Lower"
                    else:
                        continue
                    payout_multiplier = _parse_float(
                        option.get("payout_multiplier")
                        or ((option.get("odds") or {}).get("fantasy") or {}).get("multiplier")
                    )
                    entries.append(
                        UnderdogBoardEntry(
                            player_name=player_name,
                            market_key=market_key,
                            market_label=MARKET_LABEL_BY_KEY[market_key],
                            line_score=line_score,
                            opponent_abbr=opponent_abbr,
                            start_time=start_time,
                            selection_key=selection_key,
                            selection_label=selection_label,
                            payout_multiplier=payout_multiplier,
                            line_type=str(row.get("line_type") or "").strip() or None,
                            sport=sport,
                        )
                    )
        return entries

    @lru_cache(maxsize=2)
    def _cached_board_entries(self, sport: str = "nba") -> tuple[UnderdogBoardEntry, ...]:
        board = self._fetch_board(sport)
        return tuple(
            sorted(
                board,
                key=lambda entry: (
                    _parse_datetime(entry.start_time) or datetime.min,
                    entry.player_name,
                    entry.market_label,
                    entry.selection_label or "",
                    entry.line_score,
                ),
            )
        )

    def fetch_board_entries(self, sport: str = "nba") -> list[UnderdogBoardEntry]:
        return list(self._cached_board_entries(sport))

    def _line_group_rank(self, candidates: list[tuple[int, UnderdogBoardEntry]]) -> tuple[int, int, int, float, float, datetime]:
        best_context_score = max((score for score, _ in candidates), default=0)
        selections = {entry.selection_key for _, entry in candidates if entry.selection_key}
        has_two_way_market = int({"more", "less"}.issubset(selections))
        has_balanced_line = int(any(entry.line_type == "balanced" for _, entry in candidates))
        available_multipliers = [entry.payout_multiplier for _, entry in candidates if entry.payout_multiplier is not None]
        average_multiplier = (sum(available_multipliers) / len(available_multipliers)) if available_multipliers else 0.0
        multiplier_balance_score = -abs(average_multiplier - 1.0) if available_multipliers else -999.0
        latest_start = max((_parse_datetime(entry.start_time) or datetime.min for _, entry in candidates), default=datetime.min)
        return (
            best_context_score,
            has_two_way_market,
            has_balanced_line,
            multiplier_balance_score,
            -abs(candidates[0][1].line_score) if candidates else 0.0,
            latest_start,
        )

    def fetch_player_lines(
        self,
        *,
        player_name: str,
        opponent_abbr: str | None = None,
        game_date: str | None = None,
        sport: str = "nba",
    ) -> dict[str, float]:
        normalized_target = _normalize_name(player_name)
        matched_lines: dict[str, dict[float, list[tuple[int, UnderdogBoardEntry]]]] = {}

        for entry in self.fetch_board_entries(sport):
            if _normalize_name(entry.player_name) != normalized_target:
                continue

            score = 0
            if opponent_abbr and opponent_abbr.upper() == entry.opponent_abbr:
                score += 3
            entry_start = _parse_datetime(entry.start_time)
            if game_date and entry_start and entry_start.date().isoformat() == game_date:
                score += 2

            matched_lines.setdefault(entry.market_key, {}).setdefault(entry.line_score, []).append((score, entry))

        selected_lines: dict[str, float] = {}
        for market_key, line_groups in matched_lines.items():
            ranked_groups = sorted(
                line_groups.items(),
                key=lambda item: self._line_group_rank(item[1]),
                reverse=True,
            )
            if ranked_groups:
                selected_lines[market_key] = ranked_groups[0][0]
        return selected_lines
