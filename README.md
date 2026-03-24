# NBA Flask Predictor

This project serves NBA player stat predictions for points, assists, and rebounds with Flask.

## What changed

The app now has:

- environment-based RapidAPI configuration instead of hardcoded secrets
- centralized API access in `api_client.py`
- shared feature engineering in `features.py`
- cached model loading with metadata support in `modeling.py`
- a training and evaluation entry point in `train_models.py`
- an `nba_api` ingestion pipeline in `data_ingest.py`

## Run the app

1. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy the local env file and add any keys you want to use:

```bash
cp .env.example .env
```

Important `.env` values:

```bash
RAPIDAPI_KEY=your_key_here
PARLAYPLAY_COOKIE=your_browser_cookie_here
PARLAYPLAY_REFERER=https://parlayplay.io/
PARLAYPLAY_USER_AGENT=Mozilla/5.0
NBA_SEASON_START_YEAR=2025
FLASK_DEBUG=true
```

3. Start Flask:

```bash
python3 app.py
```

## Build the training dataset

Install dependencies, then pull seasons of player game logs from `nba_api`:

```bash
python3 data_ingest.py --seasons 2023-24 2024-25 --output data/player_game_logs.csv
```

The generated CSV follows this schema:

```text
game_date, season, season_type, game_id, player_id, player_name, team_id, team_abbr,
opponent_team_id, opponent_abbr, home, win, minutes, starter, rest_days, is_back_to_back,
points, assists, rebounds, off_rebounds, def_rebounds, fgm, fga, fg_pct, fg3m, fg3a,
three_pt_pct, ftm, fta, ft_pct, turnovers, steals, blocks, personal_fouls, plus_minus,
usage_rate, true_shooting_pct, player_pace, player_off_rating, player_def_rating,
team_pace, team_off_rating, team_def_rating, opp_pace, opp_off_rating, opp_def_rating,
line_points, line_assists, line_rebounds, closing_over_price, closing_under_price
```

`data_ingest.py` wires in:

- opponent team context
- rest days and back-to-backs
- minutes and a starter proxy
- player usage / true-shooting / pace / ratings
- team and opponent pace / ratings

The betting-line columns are included as placeholders so you can enrich the CSV later with props and closing prices.

## Train stronger models

`train_models.py` expects a game-by-game CSV with at least:

- `game_date`
- `player_id`
- `points`
- `assists`
- `rebounds`
- `minutes`
- `home`
- `rest_days`
- `is_back_to_back`

Train and write artifacts to the project directory:

```bash
python3 train_models.py --dataset data/player_game_logs.csv --output-dir .
```

This writes:

- `model_points.pkl`
- `model_assists.pkl`
- `model_rebounds.pkl`
- `model_metadata.json`

The training pipeline is tuned toward prop-style regression by:

- using rolling 3/5/10-game and season-average features
- including opponent pace/defense and rest context
- weighting more recent games more heavily
- selecting from target-specific LightGBM parameter sets
- reporting `within_n` tolerance hit rates in addition to `MAE`, `RMSE`, and `R²`
- optionally computing over/under hit rate when prop lines are present in the CSV

## What elite looks like

For regression, evaluate with `MAE`, `RMSE`, and `R²`, not "accuracy".

Strong targets:

- points: `MAE` near `3-4`
- assists: `MAE` near `1-2`
- rebounds: `MAE` near `1-2`
- `R²` above `0.5`

To push toward that range, your dataset needs opponent context, pace, minutes expectation, and rest/injury context in addition to recent form.

## Important note for the live app

The Flask UI now accepts an optional opponent abbreviation and game date on the search results screen. When you provide them, the app builds matchup-aware inference context for the upcoming game.

Example:

```text
Opponent: BOS
Game date: 2026-03-15
```

Notes:

- if you leave them blank, the app falls back to a player-only baseline
- if `nba_api` can load current team advanced stats, opponent pace/defense are filled automatically
- the strongest live predictions will come after retraining on the new CSV schema and then using opponent/date every time
