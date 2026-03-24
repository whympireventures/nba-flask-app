from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

import requests

from config import settings


STAT_NAME_MAP = {
    "points": "Points",
    "assists": "Assists",
    "rebounds": "Rebounds",
    "pts+reb": "Points + Rebounds",
    "pts+ast": "Points + Assists",
    "ast+reb": "Assists + Rebounds",
    "pts+reb+ast": "PRA",
    "3-pt made": None,
    "blocked shots": None,
    "steals": None,
    "turnovers": None,
}

MARKET_KEY_BY_LABEL = {
    "Points": "line_points",
    "Assists": "line_assists",
    "Rebounds": "line_rebounds",
    "Points + Rebounds": "line_points_rebounds",
    "Points + Assists": "line_points_assists",
    "Assists + Rebounds": "line_assists_rebounds",
}

MARKET_LABEL_BY_KEY = {v: k for k, v in MARKET_KEY_BY_LABEL.items()}


def _normalize_name(value: str) -> str:
    return " ".join(str(value).lower().replace(".", "").split())


def _normalize_stat(value: str) -> str:
    return str(value).strip().lower()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class PrizePicksBoardEntry:
    player_name: str
    market_key: str
    market_label: str
    line_score: float
    opponent_abbr: str
    start_time: str | None
    sport: str = "nba"


class PrizePicksProviderError(RuntimeError):
    pass


class PrizePicksClient:
    _HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": "https://app.prizepicks.com/",
        "Origin": "https://app.prizepicks.com",
        "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "Connection": "keep-alive",
    }

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(self._HEADERS)

    def is_configured(self) -> bool:
        return True

    def configuration_message(self) -> str:
        return ""

    def provider_label(self) -> str:
        return "PrizePicks"

    def _fetch_raw(self, league_id: str) -> dict[str, Any]:
        base = settings.prizepicks_api_base.rstrip("/")
        url = f"{base}/projections"
        params = {
            "league_id": league_id,
            "per_page": 250,
            "single_stat": "true",
            "in_range": "true",
        }
        resp = self.session.get(url, params=params, timeout=settings.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise PrizePicksProviderError(str(data["error"]))
        return data

    def _parse_board(self, data: dict[str, Any], sport: str = "nba") -> list[PrizePicksBoardEntry]:
        projections = data.get("data", [])
        included = {item["id"]: item for item in data.get("included", [])}

        entries: list[PrizePicksBoardEntry] = []
        for proj in projections:
            if not isinstance(proj, dict):
                continue
            attrs = proj.get("attributes", {})

            raw_stat = _normalize_stat(attrs.get("stat_display_name", ""))
            market_label = STAT_NAME_MAP.get(raw_stat)
            if not market_label:
                continue
            market_key = MARKET_KEY_BY_LABEL.get(market_label)
            if not market_key:
                continue

            line = attrs.get("line_score")
            if line is None:
                continue
            try:
                line = float(line)
            except (TypeError, ValueError):
                continue

            player_id = (
                proj.get("relationships", {})
                .get("new_player", {})
                .get("data") or {}
            ).get("id", "")
            player = included.get(player_id, {})
            pattrs = player.get("attributes", {})
            name = pattrs.get("display_name") or pattrs.get("name", "")
            if not name:
                continue

            opponent_abbr = str(attrs.get("description", "")).upper().strip()
            start_time = attrs.get("start_time") or attrs.get("board_time")

            entries.append(PrizePicksBoardEntry(
                player_name=name,
                market_key=market_key,
                market_label=market_label,
                line_score=line,
                opponent_abbr=opponent_abbr,
                start_time=start_time,
                sport=sport,
            ))

        return sorted(entries, key=lambda e: (
            _parse_datetime(e.start_time) or datetime.min,
            e.player_name,
            e.market_label,
        ))

    @lru_cache(maxsize=2)
    def _cached_board_entries(self, sport: str = "nba") -> tuple[PrizePicksBoardEntry, ...]:
        league_id = settings.prizepicks_ncaab_league_id if sport == "ncaab" else settings.prizepicks_nba_league_id
        last_exc = None
        for attempt in range(3):
            try:
                data = self._fetch_raw(league_id)
                return tuple(self._parse_board(data, sport))
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                else:
                    raise PrizePicksProviderError(f"PrizePicks API error: {exc}") from exc
            except requests.RequestException as exc:
                raise PrizePicksProviderError(f"PrizePicks request failed: {exc}") from exc
        raise PrizePicksProviderError("PrizePicks rate limited after 3 attempts") from last_exc

    def fetch_board_entries(self, sport: str = "nba") -> list[PrizePicksBoardEntry]:
        return list(self._cached_board_entries(sport))

    def fetch_player_lines(
        self,
        *,
        player_name: str,
        opponent_abbr: str | None = None,
        game_date: str | None = None,
        sport: str = "nba",
    ) -> dict[str, float]:
        normalized_target = _normalize_name(player_name)
        matched: dict[str, tuple[int, float, datetime | None]] = {}

        for entry in self.fetch_board_entries(sport):
            if _normalize_name(entry.player_name) != normalized_target:
                continue

            score = 0
            if opponent_abbr and opponent_abbr.upper() == entry.opponent_abbr:
                score += 3
            entry_start = _parse_datetime(entry.start_time)
            if game_date and entry_start and entry_start.date().isoformat() == game_date:
                score += 2

            current = matched.get(entry.market_key)
            candidate = (score, entry.line_score, entry_start)
            if current is None or score > current[0]:
                matched[entry.market_key] = candidate

        return {k: v for k, (_, v, _) in matched.items()}
