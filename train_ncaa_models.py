"""
NCAAB model training script.
Scrapes game logs from Sports Reference CBB, builds training rows using
ncaa_features.build_feature_row, trains LightGBM regressors, saves pkl files.

Usage:
    python train_ncaa_models.py
"""
from __future__ import annotations

import pickle
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from ncaa_features import build_feature_row, sort_games
from underdog_client import UnderdogClient
from config import settings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEASONS = [2023, 2024, 2025]
MIN_GAMES_FOR_TRAINING = 8   # skip players with fewer games
MIN_HISTORY_GAMES = 5        # minimum prior games to build a feature row
RATE_LIMIT_SECS = 3.5        # seconds between SR requests
TARGETS = ["points", "rebounds", "assists", "minutes"]
MODEL_DIR = settings.model_dir
SR_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Sports Reference scraper
# ---------------------------------------------------------------------------

def _name_to_slug(name: str) -> str:
    """Convert 'Cooper Flagg' → 'cooper-flagg-1'."""
    parts = re.sub(r"[^a-z ]", "", name.lower().strip()).split()
    if len(parts) < 2:
        return ""
    return f"{parts[-1][:5]}{parts[0][:2]}-01"


def _search_sr_slug(name: str, session: requests.Session) -> str | None:
    """Search Sports Reference for the player's URL slug."""
    url = "https://www.sports-reference.com/cbb/search/search.fcgi"
    try:
        r = session.get(url, params={"search": name}, timeout=10, allow_redirects=True)
        time.sleep(RATE_LIMIT_SECS)
        if r.status_code != 200:
            return None
        # If redirected directly to a player page
        if "/cbb/players/" in r.url:
            match = re.search(r"/cbb/players/([^/\.]+)", r.url)
            if match:
                return match.group(1)
        # Parse search results
        soup = BeautifulSoup(r.text, "html.parser")
        for link in soup.select("div#players a, .search-item-name a, #player_search a"):
            href = link.get("href", "")
            match = re.search(r"/cbb/players/([^/\.]+)", href)
            if match:
                return match.group(1)
    except Exception:
        pass
    return None


def _fetch_gamelog_html(slug: str, season: int, session: requests.Session) -> str | None:
    url = f"https://www.sports-reference.com/cbb/players/{slug}/gamelog/{season}"
    try:
        r = session.get(url, timeout=15)
        time.sleep(RATE_LIMIT_SECS)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def _parse_gamelog_html(html: str, player_name: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "player_game_log"}) or soup.find("table", {"id": "pgl_basic"})
    if not table:
        return []

    rows = []
    for tr in table.select("tbody tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        cells = {td.get("data-stat"): td.get_text(strip=True) for td in tr.find_all(["td", "th"])}
        date_str = cells.get("date") or cells.get("date_game", "")
        if not date_str or date_str in ("", "Date"):
            continue
        if cells.get("reason"):  # DNP row
            continue

        def sf(key):
            v = cells.get(key, "")
            if v in ("", "--", "Did Not Play", "Inactive", "Not With Team", "Suspended"):
                return 0.0
            try:
                return float(v.replace("%", "").replace("*", ""))
            except ValueError:
                return 0.0

        minutes_raw = cells.get("mp", "0")
        if ":" in str(minutes_raw):
            try:
                m, s = minutes_raw.split(":")
                minutes = float(m) + float(s) / 60.0
            except Exception:
                minutes = 0.0
        else:
            try:
                minutes = float(str(minutes_raw).strip()) if minutes_raw else 0.0
            except ValueError:
                minutes = 0.0

        if minutes <= 0 and sf("pts") == 0:
            continue

        opp = str(cells.get("opp_name_abbr") or cells.get("opp_id") or cells.get("opp", "")).upper().strip()

        game_num = cells.get("player_game_num_career") or cells.get("ranker") or cells.get("game_season", "0")
        # SR stores pct as decimals like .400 — multiply to get percentage
        def pct(key):
            v = sf(key)
            return v * 100.0 if 0.0 < v <= 1.0 else v

        rows.append({
            "game": {"id": game_num, "date": date_str},
            "player": {"id": player_name.lower()},
            "opponent_abbr": opp,
            "opponent_name": opp,
            "points": sf("pts"),
            "totReb": sf("trb"),
            "assists": sf("ast"),
            "turnovers": sf("tov"),
            "fgm": sf("fg"),
            "fga": sf("fga"),
            "fgp": pct("fg_pct"),
            "tpm": sf("fg3"),
            "tpa": sf("fg3a"),
            "tpp": pct("fg3_pct"),
            "min": minutes,
        })
    return rows


def fetch_player_gamelogs(
    player_name: str,
    session: requests.Session,
    slug_cache: dict[str, str | None],
) -> list[dict]:
    """Fetch all available game logs for a player across SEASONS."""
    if player_name not in slug_cache:
        slug = _search_sr_slug(player_name, session)
        if not slug:
            # fallback: try constructed slug variants
            parts = re.sub(r"[^a-z ]", "", player_name.lower()).split()
            if len(parts) >= 2:
                slug = f"{parts[-1][:5]}{parts[0][:2]}-01"
            else:
                slug = None
        slug_cache[player_name] = slug
        print(f"  slug for {player_name}: {slug}")

    slug = slug_cache[player_name]
    if not slug:
        return []

    all_logs: list[dict] = []
    for season in SEASONS:
        html = _fetch_gamelog_html(slug, season, session)
        if html:
            logs = _parse_gamelog_html(html, player_name)
            all_logs.extend(logs)
            if logs:
                print(f"    {player_name} {season}: {len(logs)} games")

    return all_logs


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------

def build_training_rows(player_name: str, game_logs: list[dict]) -> list[dict]:
    """
    For each game (after MIN_HISTORY_GAMES), compute features from prior games
    and use the actual game stats as labels.
    """
    sorted_logs = sort_games(game_logs)
    sorted_logs = list(reversed(sorted_logs))  # oldest first for rolling window

    rows = []
    for i in range(MIN_HISTORY_GAMES, len(sorted_logs)):
        history = sorted_logs[:i]           # all games before this one
        target_game = sorted_logs[i]        # this game is the label

        try:
            features = build_feature_row(history)
        except Exception:
            continue

        features["target_points"] = target_game.get("points", 0.0)
        features["target_rebounds"] = float(target_game.get("totReb", 0.0))
        features["target_assists"] = target_game.get("assists", 0.0)
        features["target_minutes"] = float(target_game.get("min", target_game.get("minutes", 0.0)))
        features["player_name"] = player_name
        rows.append(features)

    return rows


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

LGBM_PARAMS = {
    "points":   {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 10},
    "rebounds": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 10},
    "assists":  {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 10},
    "minutes":  {"n_estimators": 300, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 10},
}


def train_and_save(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if not c.startswith("target_") and c != "player_name"]
    X = df[feature_cols].fillna(0.0)

    for target in TARGETS:
        label_col = f"target_{target}"
        if label_col not in df.columns:
            continue
        y = df[label_col]

        params = LGBM_PARAMS.get(target, {})
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("lgbm", LGBMRegressor(random_state=42, verbose=-1, **params)),
        ])
        model.fit(X, y)
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        print(f"  {target}: MAE={mae:.3f} on {len(y)} rows")

        out_path = Path(MODEL_DIR) / f"ncaa_model_{target}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== NCAAB Model Training ===")

    # Get player list from Underdog NCAAB board
    print("\nFetching Underdog NCAAB board players...")
    player_names = []
    try:
        entries = UnderdogClient().fetch_board_entries(sport="ncaab")
        player_names = list({e.player_name for e in entries})
        print(f"Found {len(player_names)} players from Underdog board.")
    except Exception as exc:
        print(f"Could not fetch Underdog board ({exc}), using fallback player list.")

    if not player_names:
        # Fallback: top NCAAB players by prop volume this season
        player_names = [
            "Cooper Flagg", "Dylan Harper", "VJ Edgecombe", "Ace Bailey",
            "Tre Johnson", "Boogie Fland", "Kon Knueppel", "Nique Clifford",
            "Hunter Dickinson", "RJ Davis", "Zek Nnaji", "Tyrese Proctor",
            "Caleb Love", "Cody Williams", "Ja'Kobe Walter", "Matas Buzelis",
            "Yves Missi", "Isaiah Collier", "Stephon Castle", "Devin Carter",
            "Rob Dillingham", "Dalton Knecht", "Kyle Filipowski", "Armando Bacot",
            "Zach Edey", "Mark Sears", "Cam Spencer", "Tyler Kolek",
            "Ryan Kalkbrenner", "Johni Broome", "Walter Clayton Jr", "Hunter Sallis",
            "Jaeden Zackery", "Javon Small", "Taison Chatman", "Darius Adams",
            "Boo Buie", "Chase Hunter", "Dallan Coleman", "Tamin Lipsey",
            "Keshawn Murphy", "Sion James", "Jordan Dingle", "Jalen Pickett",
            "Marcus Sasser", "Jarace Walker", "Cason Wallace", "Nick Smith Jr",
            "Anthony Black", "Bilal Coulibaly",
        ]
        print(f"Using fallback list of {len(player_names)} players.")

    print(f"Found {len(player_names)} unique players on the board.\n")

    session = requests.Session()
    session.headers.update(SR_HEADERS)

    slug_cache: dict[str, str | None] = {}
    all_rows: list[dict] = []

    for i, name in enumerate(player_names, 1):
        print(f"[{i}/{len(player_names)}] {name}")
        try:
            logs = fetch_player_gamelogs(name, session, slug_cache)
            if len(logs) < MIN_GAMES_FOR_TRAINING:
                print(f"  skipped — only {len(logs)} games")
                continue
            rows = build_training_rows(name, logs)
            all_rows.extend(rows)
            print(f"  → {len(rows)} training rows")
        except Exception as exc:
            print(f"  error: {exc}")

    if not all_rows:
        print("\nNo training data collected. Exiting.")
        return

    print(f"\nTotal training rows: {len(all_rows)}")
    df = pd.DataFrame(all_rows)

    print("\nTraining models...")
    train_and_save(df)
    print("\nDone. Models saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
