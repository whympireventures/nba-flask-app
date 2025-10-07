
"""
v2_build_dataset.py — Build a multi-season NBA game-log CSV for model training.

Usage (examples):
  python v2_build_dataset.py --seasons 2022,2023,2024 --out data/game_logs_2022_2024.csv
  python v2_build_dataset.py --seasons 2024 --limit_players_per_team 8

Requirements:
  - requests, python-dotenv, tqdm, tenacity
  - .env with RAPIDAPI_KEY and (optionally) NBA_RAPIDAPI_HOST

Notes:
  - Uses RapidAPI host: api-nba-v1.p.rapidapi.com by default
  - Resilient with retries/backoff
  - Produces columns expected by v2_train_models.py
"""

import os
import csv
import time
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
NBA_HOST = os.getenv("NBA_RAPIDAPI_HOST", "api-nba-v1.p.rapidapi.com")

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY or "",
    "x-rapidapi-host": NBA_HOST,
}

def _check_key():
    if not RAPIDAPI_KEY:
        raise SystemExit("RAPIDAPI_KEY missing. Copy .env.example -> .env and set RAPIDAPI_KEY.")

def _url(path: str) -> str:
    return f"https://{NBA_HOST}{path}" if not path.startswith("http") else path

class HttpError(Exception):
    pass

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(HttpError))
def _get(path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    resp = requests.get(_url(path), headers=HEADERS, params=params or {}, timeout=20)
    if resp.status_code >= 400:
        raise HttpError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except Exception as e:
        raise HttpError(f"Invalid JSON: {e}")

def fetch_teams(league: str = "standard") -> List[Dict[str, Any]]:
    data = _get("/teams", params={"league": league})
    return data.get("response", [])

def fetch_players(team_id: int, season: str) -> List[Dict[str, Any]]:
    # API-NBA paginates players; iterate until no results
    players = []
    page = 1
    while True:
        data = _get("/players", params={"team": team_id, "season": season, "page": page})
        res = data.get("response", [])
        if not res:
            break
        players.extend(res)
        page += 1
        if page > 50:  # safety
            break
    return players

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _to_minutes(x):
    # "MM:SS" -> float minutes
    if isinstance(x, str) and ":" in x:
        mm, ss = x.split(":")
        try:
            return float(mm) + float(ss)/60.0
        except Exception:
            return None
    return _to_float(x)

def fetch_player_stats(player_id: int, season: str) -> List[Dict[str, Any]]:
    data = _get("/players/statistics", params={"id": player_id, "season": season})
    return data.get("response", [])

def rows_from_stats(stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for g in stats:
        game = g.get("game", {}) or {}
        teams = game.get("teams", {}) or {}
        t_home = (teams.get("home") or {}).get("id")
        t_away = (teams.get("visitors") or {}).get("id")
        team_id = (g.get("team") or {}).get("id")
        # opp is the other team if we can infer
        opp_team_id = None
        if team_id and t_home and t_away:
            opp_team_id = t_away if team_id == t_home else t_home

        # home flag
        home = None
        if team_id and t_home:
            home = 1 if team_id == t_home else 0

        rows.append({
            "player_id": (g.get("player") or {}).get("id"),
            "game_date": game.get("date"),
            "team_id": team_id,
            "opp_team_id": opp_team_id,
            "home": home,
            "minutes": _to_minutes(g.get("min")),
            "pts": _to_float(g.get("points")),
            "ast": _to_float(g.get("assists")),
            "reb": _to_float(g.get("totReb")),
            "fga": _to_float(g.get("fga")),
            "fgm": _to_float(g.get("fgm")),
            "fg3a": _to_float(g.get("tpa")),
            "fg3m": _to_float(g.get("tpm")),
            "fta": _to_float(g.get("fta")),
            "ftm": _to_float(g.get("ftm")),
            "tov": _to_float(g.get("turnovers")),
        })
    return rows

def build_csv(seasons: List[str], out_path: Path, limit_players_per_team: Optional[int] = None, sleep_sec: float = 0.4):
    _check_key()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["player_id","game_date","team_id","opp_team_id","home",
              "minutes","pts","ast","reb","fga","fgm","fg3a","fg3m","fta","ftm","tov"]
    wrote_header = False

    teams = fetch_teams(league="standard")
    teams = [t for t in teams if t.get("nbaFranchise") and t.get("leagues", {}).get("standard", {}).get("conference")]
    teams_sorted = sorted(teams, key=lambda t: t.get("name",""))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        wrote_header = True

        for season in seasons:
            print(f"\n=== Season {season} ===")
            for t in tqdm(teams_sorted, desc=f"Teams {season}"):
                team_id = t.get("id")
                if not team_id:
                    continue
                players = fetch_players(team_id=team_id, season=season)
                if limit_players_per_team:
                    players = players[:limit_players_per_team]

                for p in players:
                    pid = (p.get("id") or p.get("player",{}).get("id"))
                    if not pid:
                        continue
                    stats = fetch_player_stats(player_id=pid, season=season)
                    rows = rows_from_stats(stats)
                    for r in rows:
                        writer.writerow(r)
                    time.sleep(sleep_sec)  # be nice to the API

    print(f"\nSaved CSV -> {out_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=str, default="2024",
                    help="Comma-separated list, e.g. 2022,2023,2024")
    ap.add_argument("--out", type=str, default="data/game_logs.csv")
    ap.add_argument("--limit_players_per_team", type=int, default=None,
                    help="Debugging helper to reduce cost/time per run")
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds to sleep between player calls")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    out_path = Path(args.out)
    build_csv(seasons=seasons, out_path=out_path,
              limit_players_per_team=args.limit_players_per_team,
              sleep_sec=args.sleep)
