import pickle, pandas as pd, numpy as np
from flask import current_app
from services.api_nba import player_game_stats

def _ewma(vals, alpha=0.6):
    if not vals: return 0.0
    out = vals[0]
    for v in vals[1:]:
        out = alpha*v + (1-alpha)*out
    return float(out)

def predict_player_stats(player_id):
    season = current_app.config["DEFAULT_SEASON"]
    data = player_game_stats(player_id, season)
    games = (data or {}).get("response", [])
    games = sorted(games, key=lambda g: g.get("game",{}).get("id",0), reverse=True)[:10]
    if not games: return 0.0, 0.0, 0.0

    def num(x):
        try: return float(x)
        except: return 0.0

    pts = [num(g.get("points",0)) for g in games]
    ast = [num(g.get("assists",0)) for g in games]
    reb = [num(g.get("totReb",0)) for g in games]

    feat = pd.DataFrame([{
        "ewma_pts": _ewma(pts), "ewma_ast": _ewma(ast), "ewma_reb": _ewma(reb),
        "avg_pts": np.mean(pts) if pts else 0.0,
        "avg_ast": np.mean(ast) if ast else 0.0,
        "avg_reb": np.mean(reb) if reb else 0.0
    }])

    try:
        with open('model_points.pkl','rb') as f: mp = pickle.load(f)
        with open('model_assists.pkl','rb') as f: ma = pickle.load(f)
        with open('model_rebounds.pkl','rb') as f: mr = pickle.load(f)
        return float(mp.predict(feat)[0]), float(ma.predict(feat)[0]), float(mr.predict(feat)[0])
    except Exception:
        return feat.ewma_pts.iloc[0], feat.ewma_ast.iloc[0], feat.ewma_reb.iloc[0]
