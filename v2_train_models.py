
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool

# === CONFIG ===
DATA_CSV = "PATH/TO/your_player_game_logs.csv"  # <-- CHANGE ME
OUT_DIR = Path("models_v2")
OUT_DIR.mkdir(exist_ok=True, parents=True)

TARGETS = ["pts", "ast", "reb"]    # points, assists, rebounds
ROLLS = [3, 5, 10]                  # rolling windows

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"]).copy()
    # Basic usage proxy
    df["usage_proxy"] = (df["fga"] + 0.44*df["fta"] + df.get("tov", 0)).astype(float)
    # Rolling windows per player
    feat_cols = ["minutes","fga","fg3a","fta","usage_proxy","pts","ast","reb","tov"]
    for col in feat_cols:
        for w in ROLLS:
            df[f"{col}_rmean_{w}"] = df.groupby("player_id")[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            df[f"{col}_rstd_{w}"]  = df.groupby("player_id")[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).std())
    # Rest days
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["prev_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - df["prev_date"]).dt.days.clip(lower=0).fillna(3)

    # Keep a clean feature set
    feature_cols = [c for c in df.columns if any(p in c for p in ["_rmean_","_rstd_"]) or c in ["rest_days","home","pace","team_off_rating","opp_def_rating","minutes_rmean_3"]]
    # Ensure no targets leak
    feature_cols = [c for c in feature_cols if c not in TARGETS]
    return df, feature_cols

def train_one(df, feature_cols, target):
    # Time-based split (last ~20% as validation)
    df = df.dropna(subset=feature_cols+[target]).copy()
    df = df.sort_values("game_date")

    split_idx = int(len(df)*0.8)
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target]

    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.05,
        iterations=1500,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        l2_leaf_reg=5.0,
        verbose=200
    )
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    rmse = mean_squared_error(y_valid, pred, squared=False)
    print(f"[{target}] MAE={mae:.3f}  RMSE={rmse:.3f}  n_valid={len(y_valid)}")

    model.save_model(str(OUT_DIR/f"model_{target}.cbm"))
    return model, {"mae": mae, "rmse": rmse}

def main():
    df = pd.read_csv(DATA_CSV)
    # basic cleaning
    needed = ["player_id","game_date","minutes","pts","ast","reb","fga","fg3a","fta","tov","home"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Your CSV is missing required columns: {missing}")

    df, feature_cols = add_features(df)
    print(f"Using {len(feature_cols)} feature columns")

    metrics = {}
    for t in TARGETS:
        _, m = train_one(df, feature_cols, t)
        metrics[t] = m

    print("Final metrics:", metrics)

if __name__ == "__main__":
    main()
