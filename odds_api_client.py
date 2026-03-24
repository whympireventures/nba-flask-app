from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests

from config import settings


MARKET_MAP = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_points_rebounds": "Points + Rebounds",
    "player_points_assists": "Points + Assists",
    "player_rebounds_assists": "Assists + Rebounds",
    "player_points_rebounds_assists": "PRA",
}

BOOKMAKERS = "draftkings,fanduel,betmgm,betrivers,williamhill_us"


@dataclass(frozen=True)
class OddsApiEntry:
    player_name: str
    market_label: str
    line: float
    book: str


class OddsApiProviderError(RuntimeError):
    pass


class OddsApiClient:
    _BASE = "https://api.the-odds-api.com/v4"

    def __init__(self) -> None:
        self.api_key = settings.odds_api_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "Mozilla/5.0"})

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict) -> Any:
        params["apiKey"] = self.api_key
        r = self.session.get(f"{self._BASE}{path}", params=params, timeout=settings.request_timeout)
        r.raise_for_status()
        return r.json()

    def _fetch_events(self) -> list[dict]:
        try:
            return self._get("/sports/basketball_nba/events", {})
        except requests.RequestException as exc:
            raise OddsApiProviderError(f"Odds API events error: {exc}") from exc

    def _fetch_event_odds(self, event_id: str) -> dict:
        params = {
            "regions": "us",
            "markets": ",".join(MARKET_MAP.keys()),
            "bookmakers": BOOKMAKERS,
            "oddsFormat": "american",
        }
        try:
            return self._get(f"/sports/basketball_nba/events/{event_id}/odds", params)
        except requests.RequestException:
            return {}

    @lru_cache(maxsize=1)
    def _cached_entries(self) -> tuple[OddsApiEntry, ...]:
        if not self.is_configured():
            raise OddsApiProviderError("Set ODDS_API_KEY to enable multi-book line shopping.")
        events = self._fetch_events()
        entries: list[OddsApiEntry] = []
        for event in events:
            event_id = event.get("id", "")
            if not event_id:
                continue
            odds_data = self._fetch_event_odds(event_id)
            for book in odds_data.get("bookmakers", []):
                book_name = book.get("title", book.get("key", ""))
                for market in book.get("markets", []):
                    market_label = MARKET_MAP.get(market.get("key", ""))
                    if not market_label:
                        continue
                    seen: dict[str, float] = {}
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") != "Over":
                            continue
                        player = str(outcome.get("description", "")).strip()
                        point = outcome.get("point")
                        if not player or point is None:
                            continue
                        try:
                            seen[player] = float(point)
                        except (TypeError, ValueError):
                            continue
                    for player, line in seen.items():
                        entries.append(OddsApiEntry(
                            player_name=player,
                            market_label=market_label,
                            line=line,
                            book=book_name,
                        ))
        return tuple(entries)

    def fetch_entries(self) -> list[OddsApiEntry]:
        return list(self._cached_entries())

    def build_line_map(self) -> dict[tuple[str, str], dict[str, float]]:
        """Returns {(norm_name, market): {book_name: line}} for all players."""
        def _norm(s: str) -> str:
            return " ".join(str(s).lower().replace(".", "").split())

        result: dict[tuple[str, str], dict[str, float]] = {}
        for entry in self.fetch_entries():
            key = (_norm(entry.player_name), entry.market_label)
            result.setdefault(key, {})[entry.book] = entry.line
        return result
