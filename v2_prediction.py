
import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from catboost import CatBoostRegressor

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
NBA_HOST = os.getenv("NBA_RAPIDAPI_HOST", "api-nba-v1.p.rapidapi.com")
DEFAULT_SEASON = os.getenv("DEFAULT_SEASON", "2024")

MODELS_DIR = Path("models_v2")
MODEL_PATHS = {
    "pts": MODELS_DIR/"model_points.cbm",
    "ast": MODELS_DIR/"model_assists.cbm",
    "reb": MODELS_DIR/"model_rebounds.cbm",
}

def _headers():
    return {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": NBA_HOST
    }

def fetch_player_last_games(player_id: int, season: str = None, limit: int = 12) -> pd.DataFrame:
    """Pull last `limit` games for player from the stats endpoint and return as DataFrame."""
    season = str(season or DEFAULT_SEASON)
    url = f"https://{NBA_HOST}/players/statistics"
    params = {"id": player_id, "season": season}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("response", [])
    if not data:
        return pd.DataFrame()

    # Normalize
    rows = []
    for g in data:
        rows.append({
            "player_id": player_id,
            "game_date": g.get("game", {}).get("date", None),
            "minutes": _to_minutes(g.get("min")),
            "pts": _to_float(g.get("points")),
            "ast": _to_float(g.get("assists")),
            "reb": _to_float(g.get("totReb")),
            "fga": _to_float(g.get("fga")),
            "fg3a": _to_float(g.get("tpa")),
            "fta": _to_float(g.get("fta")),
            "tov": _to_float(g.get("turnovers")),
            "home": 1 if g.get("game", {}).get("arena", {}).get("name") else np.nan,  # placeholder if home/away unavailable
        })
    df = pd.DataFrame(rows)
    # keep only last N by date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values("game_date").tail(limit)
    return df.reset_index(drop=True)

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _to_minutes(x):
    # convert "MM:SS" to float minutes
    if isinstance(x, str) and ":" in x:
        mm, ss = x.split(":")
        try:
            return float(mm) + float(ss)/60.0
        except Exception:
            return np.nan
    return _to_float(x)

def build_features_from_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["player_id","game_date"]).copy()
    df["usage_proxy"] = (df["fga"] + 0.44*df["fta"] + df["tov"].fillna(0)).astype(float)
    ROLLS = [3,5,10]
    feat_cols = ["minutes","fga","fg3a","fta","usage_proxy","pts","ast","reb","tov"]
    for col in feat_cols:
        for w in ROLLS:
            df[f"{col}_rmean_{w}"] = df.groupby("player_id")[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            df[f"{col}_rstd_{w}"]  = df.groupby("player_id")[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).std())
    df["prev_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - df["prev_date"]).dt.days.clip(lower=0).fillna(3)

    feature_cols = [c for c in df.columns if any(p in c for p in ["_rmean_","_rstd_"]) or c in ["rest_days","home"]]
    latest = df.dropna(subset=feature_cols).iloc[-1:][feature_cols].copy()
    return latest

def _load_models():
    models = {}
    for k, p in MODEL_PATHS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        m = CatBoostRegressor()
        m.load_model(str(p))
        models[k] = m
    return models

def predict_player_stats(player_id: int, season: str = None):
    """Return (points, assists, rebounds) prediction for next game-like context."""
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_KEY is not set. Copy .env.example → .env and set it.")

    df_games = fetch_player_last_games(player_id, season=season, limit=12)
    if df_games.empty:
        return 0.0, 0.0, 0.0

    X = build_features_from_games(df_games)
    if X.empty:
        return 0.0, 0.0, 0.0

    models = _load_models()

    pts = float(models["pts"].predict(X)[0])
    ast = float(models["ast"].predict(X)[0])
    reb = float(models["reb"].predict(X)[0])
    return round(pts, 2), round(ast, 2), round(reb, 2)

# For manual quick test:
# if __name__ == "__main__":
#     print(predict_player_stats(player_id=125))  # e.g., Stephen Curry ID (replace)
