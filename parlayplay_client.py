from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import requests

from config import settings


MARKET_KEY_BY_LABEL = {
    "points": "line_points",
    "assists": "line_assists",
    "rebounds": "line_rebounds",
    "pts + reb": "line_points_rebounds",
    "pts + ast": "line_points_assists",
    "reb + ast": "line_assists_rebounds",
    "pts + reb + ast": "line_points_rebounds_assists",
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
SELECTION_KEY_BY_VALUE = {
    "more": "more",
    "higher": "more",
    "over": "more",
    "up": "more",
    "less": "less",
    "lower": "less",
    "under": "less",
    "down": "less",
}
SIDE_MULTIPLIER_KEYS = {
    "more": ("moreMultiplier", "higherMultiplier", "overMultiplier"),
    "less": ("lessMultiplier", "lowerMultiplier", "underMultiplier"),
}
LINE_VALUE_KEYS = ("statValue", "line", "lineScore", "score")
MULTIPLIER_KEYS = ("payoutMultiplier", "multiplier", "decimalMultiplier", "decimalOdds", "oddsMultiplier")


def _normalize_name(value: str) -> str:
    return " ".join(str(value).lower().replace(".", "").split())


def _normalize_market_label(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().replace("rebounds", "reb").replace("assists", "ast").replace("points", "pts").split())


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


def _normalize_selection(value: object) -> str | None:
    normalized = " ".join(str(value or "").strip().lower().split())
    return SELECTION_KEY_BY_VALUE.get(normalized)


def _selection_label(selection_key: str | None) -> str | None:
    if selection_key == "more":
        return "More"
    if selection_key == "less":
        return "Less"
    return None


def _extract_cookie_value(cookie_header: str, cookie_name: str) -> str:
    for chunk in cookie_header.split(";"):
        part = chunk.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip() == cookie_name:
            return value.strip()
    return ""


@dataclass(frozen=True)
class ParlayPlayBoardEntry:
    player_name: str
    market_key: str
    market_label: str
    line_score: float
    opponent_abbr: str
    start_time: str | None
    selection_key: str | None = None
    selection_label: str | None = None
    payout_multiplier: float | None = None


class ParlayPlayProviderError(RuntimeError):
    pass


class ParlayPlayClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": settings.parlayplay_accept_language,
                "Origin": settings.parlayplay_origin,
                "User-Agent": settings.parlayplay_user_agent,
                "Referer": settings.parlayplay_referer,
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        if settings.parlayplay_cookie:
            self.session.headers["Cookie"] = settings.parlayplay_cookie
            csrf_token = _extract_cookie_value(settings.parlayplay_cookie, "csrftoken")
            if csrf_token:
                self.session.headers["X-CSRFToken"] = csrf_token

    def is_configured(self) -> bool:
        return bool(settings.parlayplay_cookie)

    def configuration_message(self) -> str:
        if self.is_configured():
            return ""
        return "Set PARLAYPLAY_COOKIE to let the app call the ParlayPlay board with your approved browser session."

    def setup_hint(self) -> str:
        return (
            "Recommended env vars: PARLAYPLAY_COOKIE, PARLAYPLAY_REFERER, PARLAYPLAY_USER_AGENT, "
            "PARLAYPLAY_ORIGIN, and PARLAYPLAY_ACCEPT_LANGUAGE. Cookie and an exact browser user-agent are the most important."
        )

    def _get(self, path: str, params: dict[str, str]) -> dict:
        last_exc = None
        for attempt in range(3):
            response = self.session.get(
                f"{settings.parlayplay_api_base.rstrip('/')}{path}",
                params=params,
                timeout=settings.request_timeout,
            )
            if response.status_code == 429:
                last_exc = requests.HTTPError(response=response)
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code in (401, 403):
                raise ParlayPlayProviderError(
                    f"ParlayPlay blocked the board request. {self.configuration_message() or self.setup_hint()}"
                )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ParlayPlayProviderError("ParlayPlay returned an unexpected response shape.")
            return data
        raise ParlayPlayProviderError("ParlayPlay rate limited after 3 attempts") from last_exc

    def _extract_opponent_abbr(self, row: dict) -> str:
        match = row.get("match") or {}
        player = row.get("player") or {}
        team = player.get("team") or {}
        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}
        player_team_id = team.get("id")
        home_team_id = home_team.get("id")
        away_team_id = away_team.get("id")
        if player_team_id == home_team_id:
            return str(away_team.get("teamAbbreviation") or "").upper()
        if player_team_id == away_team_id:
            return str(home_team.get("teamAbbreviation") or "").upper()
        return str((away_team.get("teamAbbreviation") or home_team.get("teamAbbreviation") or "")).upper()

    def _extract_selection_key(self, row: dict, inherited_selection: str | None = None) -> str | None:
        for field_name in ("selection", "pickType", "pick", "direction", "side", "choice", "label", "type", "name"):
            selection_key = _normalize_selection(row.get(field_name))
            if selection_key:
                return selection_key
        return inherited_selection

    def _extract_line_value(self, row: dict) -> float | None:
        for field_name in LINE_VALUE_KEYS:
            line_value = _parse_float(row.get(field_name))
            if line_value is not None:
                return line_value
        return None

    def _extract_multiplier(self, row: dict) -> float | None:
        for field_name in MULTIPLIER_KEYS:
            multiplier = _parse_float(row.get(field_name))
            if multiplier is not None:
                return multiplier
        return None

    def _expand_line_variants(
        self,
        row: dict,
        *,
        inherited_selection: str | None = None,
    ) -> list[tuple[float, str | None, float | None]]:
        line_value = self._extract_line_value(row)
        if line_value is None:
            return []

        selection_key = self._extract_selection_key(row, inherited_selection)
        multiplier = self._extract_multiplier(row)
        variants: list[tuple[float, str | None, float | None]] = []
        if selection_key:
            variants.append((line_value, selection_key, multiplier))

        for side_key, field_names in SIDE_MULTIPLIER_KEYS.items():
            for field_name in field_names:
                side_multiplier = _parse_float(row.get(field_name))
                if side_multiplier is not None:
                    variants.append((line_value, side_key, side_multiplier))
                    break

        if not variants:
            variants.append((line_value, None, multiplier))
        return variants

    def _collect_stat_entries(self, stat_row: dict) -> list[tuple[float, str | None, float | None]]:
        collected: list[tuple[float, str | None, float | None]] = []
        seen: set[tuple[float, str | None, float | None]] = set()

        def visit(node: object, inherited_selection: str | None = None) -> None:
            if isinstance(node, dict):
                selection_key = self._extract_selection_key(node, inherited_selection)
                for candidate in self._expand_line_variants(node, inherited_selection=selection_key):
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    collected.append(candidate)
                for key, value in node.items():
                    visit(value, _normalize_selection(key) or selection_key)
            elif isinstance(node, list):
                for item in node:
                    visit(item, inherited_selection)

        visit(stat_row)
        return collected

    def _fetch_board(self) -> list[ParlayPlayBoardEntry]:
        payload = self._get(
            settings.parlayplay_search_path,
            {
                "sport": "All",
                "league": "",
                "period": settings.parlayplay_period,
                "includeAlt": "true",
                "version": "2",
                "includeBoost": "true",
                "includeSports": "true",
            },
        )
        players = payload.get("players")
        if not isinstance(players, list):
            return []

        entries: list[ParlayPlayBoardEntry] = []
        for row in players:
            if not isinstance(row, dict):
                continue
            player = row.get("player") or {}
            stats = row.get("stats") or []
            player_name = str(player.get("fullName") or "").strip()
            if not player_name or not isinstance(stats, list):
                continue
            opponent_abbr = self._extract_opponent_abbr(row)
            start_time = (row.get("match") or {}).get("matchDate")

            for stat_row in stats:
                if not isinstance(stat_row, dict):
                    continue
                market_key = MARKET_KEY_BY_LABEL.get(_normalize_market_label(stat_row.get("challengeName")))
                if not market_key:
                    continue
                for numeric_line, selection_key, payout_multiplier in self._collect_stat_entries(stat_row):
                    entries.append(
                        ParlayPlayBoardEntry(
                            player_name=player_name,
                            market_key=market_key,
                            market_label=MARKET_LABEL_BY_KEY[market_key],
                            line_score=numeric_line,
                            opponent_abbr=opponent_abbr,
                            start_time=start_time,
                            selection_key=selection_key,
                            selection_label=_selection_label(selection_key),
                            payout_multiplier=payout_multiplier,
                        )
                    )
        return entries

    @lru_cache(maxsize=1)
    def _cached_board_entries(self) -> tuple[ParlayPlayBoardEntry, ...]:
        board = self._fetch_board()
        return tuple(
            sorted(
                board,
                key=lambda entry: (
                    _parse_datetime(entry.start_time) or datetime.min,
                    entry.player_name,
                    entry.market_label,
                    entry.selection_label or "",
                    entry.payout_multiplier or 0.0,
                    entry.line_score,
                ),
            )
        )

    def fetch_board_entries(self) -> list[ParlayPlayBoardEntry]:
        return list(self._cached_board_entries())

    def fetch_player_lines(
        self,
        *,
        player_name: str,
        opponent_abbr: str | None = None,
        game_date: str | None = None,
    ) -> dict[str, float]:
        normalized_target = _normalize_name(player_name)
        matched_lines: dict[str, dict[float, list[tuple[int, ParlayPlayBoardEntry]]]] = {}

        for entry in self.fetch_board_entries():
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

    def _line_group_rank(self, candidates: list[tuple[int, ParlayPlayBoardEntry]]) -> tuple[int, int, int, float, float, datetime]:
        best_context_score = max((score for score, _ in candidates), default=0)
        selections = {entry.selection_key for _, entry in candidates if entry.selection_key}
        has_two_way_market = int({"more", "less"}.issubset(selections))
        has_plain_line = int(any(entry.selection_key is None for _, entry in candidates))

        more_multipliers = [entry.payout_multiplier for _, entry in candidates if entry.selection_key == "more" and entry.payout_multiplier is not None]
        less_multipliers = [entry.payout_multiplier for _, entry in candidates if entry.selection_key == "less" and entry.payout_multiplier is not None]
        if more_multipliers and less_multipliers:
            average_gap_score = -abs((sum(more_multipliers) / len(more_multipliers)) - (sum(less_multipliers) / len(less_multipliers)))
        else:
            average_gap_score = -999.0

        available_multipliers = [entry.payout_multiplier for _, entry in candidates if entry.payout_multiplier is not None]
        if available_multipliers:
            average_multiplier = sum(available_multipliers) / len(available_multipliers)
            multiplier_balance_score = -abs(average_multiplier - 1.8)
        else:
            multiplier_balance_score = -999.0

        latest_start = max((_parse_datetime(entry.start_time) or datetime.min for _, entry in candidates), default=datetime.min)
        return (
            best_context_score,
            has_two_way_market,
            has_plain_line,
            average_gap_score,
            multiplier_balance_score,
            latest_start,
        )
