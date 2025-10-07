# NBA Player Stats Predictor — Upgrade Plan (v2)

This upgrade focuses on **accuracy**, **stability**, and **maintainability**.

## What’s new (high‑impact changes)
1) **Better features (rolling windows)** — last 3/5/10 game rolling means & std for core stats (minutes, FGA, 3PA, FTA, usage proxy, rebounds, assists, turnovers, plus pace if available).
2) **Time‑aware CV** — `TimeSeriesSplit` (grouped by player) to avoid leakage and overfitting.
3) **Model choice** — CatBoost (handles categorical & missing data well) with sensible hyperparams; fallback LightGBM/Sklearn if needed.
4) **Quantiles** — optional P10 / P50 / P90 for “range” predictions (good for props/over‑unders).
5) **Dynamic season** — no hard‑coded “2023”; you can set via env/config.
6) **Secrets in .env** — no keys in source; rotate the exposed one and put into `.env`.
7) **Single inference path** — clean feature builder at inference (same transforms as training).

---

## File map
- `v2_train_models.py` — end‑to‑end trainer (CSV in → 3 CatBoost models out). Plug in your dataset path and run.
- `v2_prediction.py` — drop‑in style inference that builds features from the player’s last N games pulled from the API.
- `.env.example` — put your RapidAPI key here and copy to `.env`.
- `requirements_v2.txt` — extra libs for training (CatBoost, scikit‑learn, python-dotenv).

> **Tip:** Keep your current app working while you test v2 models in parallel. When satisfied, swap imports in `app.py` from `prediction` to `v2_prediction` and update the model paths.

---

## Data you’ll need for training
- A CSV of **player game logs** (multiple seasons recommended, e.g., 2019–2025).
- Minimum columns (by row = one game):  
  `player_id, game_date, team_id, opp_team_id, home, minutes, pts, ast, reb, fga, fgm, fg3a, fg3m, fta, ftm, tov`
- Optional (boosts accuracy): `pace, team_off_rating, opp_def_rating, rest_days, started`.

Use your existing pipelines or the RapidAPI NBA endpoints to build this dataset offline, then point `DATA_CSV` in `v2_train_models.py` to it.

---

## Training quickstart
```bash
# 1) Create & activate a venv (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install training deps
pip install -r requirements_v2.txt

# 3) Put your game-log CSV somewhere and update DATA_CSV in v2_train_models.py

# 4) Train
python v2_train_models.py

# 5) Three models are saved:
#   models_v2/model_points.cbm
#   models_v2/model_assists.cbm
#   models_v2/model_rebounds.cbm
```

---

## Inference quickstart
```bash
# 1) Copy .env.example → .env and set RAPIDAPI_KEY
# 2) Place the three trained models in models_v2/
# 3) In your Flask app, replace:
#    from prediction import predict_player_stats
#   with:
#    from v2_prediction import predict_player_stats
```

---

## Evaluation
During training, the script prints MAE/RMSE on a **time-based validation fold** and logs feature importances.
Target accuracy goals (rough ballpark, will vary by data):
- Points MAE: 3.5–5.0
- Assists MAE: 1.5–2.2
- Rebounds MAE: 1.8–2.6

If you add context features (opponent defensive rating, pace, rest days, starter flag), you can tighten MAE further.

---

## Production notes
- **Cache** player last‑N games for 15–30 min to reduce API calls.
- **Rotate & store secrets** with `.env` (or real secret manager).
- **Pin model versions** in a `MODEL_VERSION` constant so you can roll back cleanly.
- **Log** final features & predictions for auditability.
