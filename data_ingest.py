from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats, playergamelogs
    from nba_api.stats.static import teams as nba_static_teams
except ImportError as exc:  # pragma: no cover - import guard for environments without nba_api
    raise SystemExit(
        "nba_api is required for data ingestion. Install dependencies from requirements.txt first."
    ) from exc


GAME_LOG_COLUMN_MAP = {
    "SEASON_YEAR": "season",
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team_abbr",
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "MATCHUP": "matchup",
    "WL": "win_loss",
    "MIN": "minutes",
    "PTS": "points",
    "AST": "assists",
    "REB": "rebounds",
    "OREB": "off_rebounds",
    "DREB": "def_rebounds",
    "FGM": "fgm",
    "FGA": "fga",
    "FG_PCT": "fg_pct",
    "FG3M": "fg3m",
    "FG3A": "fg3a",
    "FG3_PCT": "three_pt_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT_PCT": "ft_pct",
    "TOV": "turnovers",
    "STL": "steals",
    "BLK": "blocks",
    "PF": "personal_fouls",
    "PLUS_MINUS": "plus_minus",
}


TEAM_CONTEXT_COLUMN_MAP = {
    "TEAM_ID": "opponent_team_id",
    "PACE": "opp_pace",
    "OFF_RATING": "opp_off_rating",
    "DEF_RATING": "opp_def_rating",
}


PLAYER_CONTEXT_COLUMN_MAP = {
    "PLAYER_ID": "player_id",
    "USG_PCT": "usage_rate",
    "TS_PCT": "true_shooting_pct",
    "PACE": "player_pace",
    "OFF_RATING": "player_off_rating",
    "DEF_RATING": "player_def_rating",
}


FINAL_SCHEMA = [
    "game_date",
    "season",
    "season_type",
    "game_id",
    "player_id",
    "player_name",
    "team_id",
    "team_abbr",
    "opponent_team_id",
    "opponent_abbr",
    "home",
    "win",
    "minutes",
    "starter",
    "rest_days",
    "is_back_to_back",
    "points",
    "assists",
    "rebounds",
    "off_rebounds",
    "def_rebounds",
    "fgm",
    "fga",
    "fg_pct",
    "fg3m",
    "fg3a",
    "three_pt_pct",
    "ftm",
    "fta",
    "ft_pct",
    "turnovers",
    "steals",
    "blocks",
    "personal_fouls",
    "plus_minus",
    "usage_rate",
    "true_shooting_pct",
    "player_pace",
    "player_off_rating",
    "player_def_rating",
    "team_pace",
    "team_off_rating",
    "team_def_rating",
    "opp_pace",
    "opp_off_rating",
    "opp_def_rating",
    "line_points",
    "line_assists",
    "line_rebounds",
    "closing_over_price",
    "closing_under_price",
]


def _fetch_player_game_logs(season: str, season_type: str) -> pd.DataFrame:
    endpoint = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        date_from_nullable="",
        date_to_nullable="",
    )
    frame = endpoint.get_data_frames()[0]
    frame = frame.rename(columns=GAME_LOG_COLUMN_MAP)
    frame["season_type"] = season_type
    return frame


def _fetch_team_context(season: str, season_type: str) -> pd.DataFrame:
    endpoint = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    frame = endpoint.get_data_frames()[0]
    keep = [column for column in TEAM_CONTEXT_COLUMN_MAP if column in frame.columns]
    frame = frame[keep].rename(columns=TEAM_CONTEXT_COLUMN_MAP)
    team_lookup = {
        int(team["id"]): str(team["abbreviation"]).upper()
        for team in nba_static_teams.get_teams()
    }
    frame["opponent_team_id"] = pd.to_numeric(frame["opponent_team_id"], errors="coerce")
    frame["opponent_abbr"] = frame["opponent_team_id"].map(team_lookup)
    return frame


def _fetch_player_context(season: str, season_type: str) -> pd.DataFrame:
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    frame = endpoint.get_data_frames()[0]
    keep = [column for column in PLAYER_CONTEXT_COLUMN_MAP if column in frame.columns]
    frame = frame[keep].rename(columns=PLAYER_CONTEXT_COLUMN_MAP)
    return frame


def _extract_matchup_context(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["home"] = frame["matchup"].str.contains("vs.", regex=False).astype(int)
    frame["opponent_abbr"] = frame["matchup"].str.split().str[-1].fillna("").astype(str).str.upper()
    frame["win"] = (frame["win_loss"] == "W").astype(int)
    return frame


def _compute_rest_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["player_id", "game_date"]).copy()
    previous_date = frame.groupby("player_id")["game_date"].shift(1)
    frame["rest_days"] = (frame["game_date"] - previous_date).dt.days.sub(1)
    frame["rest_days"] = frame["rest_days"].clip(lower=0).fillna(3)
    frame["is_back_to_back"] = (frame["rest_days"] == 0).astype(int)
    return frame


def _compute_starter_flag(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    median_minutes = frame.groupby("player_id")["minutes"].transform("median")
    frame["starter"] = ((frame["minutes"] >= 24) | (median_minutes >= 28)).astype(int)
    return frame


def _merge_team_context(frame: pd.DataFrame, team_context: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    team_context = team_context.copy()
    frame["opponent_abbr"] = frame["opponent_abbr"].fillna("").astype(str).str.upper()
    frame["team_abbr"] = frame["team_abbr"].fillna("").astype(str).str.upper()
    team_context["opponent_abbr"] = team_context["opponent_abbr"].fillna("").astype(str).str.upper()
    enriched = frame.merge(team_context, on="opponent_abbr", how="left")
    own_team_context = team_context.rename(
        columns={
            "opponent_team_id": "team_id",
            "opponent_abbr": "team_abbr",
            "opp_pace": "team_pace",
            "opp_off_rating": "team_off_rating",
            "opp_def_rating": "team_def_rating",
        }
    )
    enriched = enriched.merge(own_team_context, on=["team_id", "team_abbr"], how="left")
    return enriched


def _ensure_final_schema(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in FINAL_SCHEMA:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[FINAL_SCHEMA].sort_values(["game_date", "player_id"]).reset_index(drop=True)


def build_dataset(seasons: list[str], season_type: str) -> pd.DataFrame:
    season_frames = []
    for season in seasons:
        game_logs = _fetch_player_game_logs(season, season_type)
        team_context = _fetch_team_context(season, season_type)
        player_context = _fetch_player_context(season, season_type)

        frame = _extract_matchup_context(game_logs)
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
        numeric_columns = [
            column
            for column in frame.columns
            if column not in {"player_name", "team_abbr", "opponent_abbr", "matchup", "win_loss", "season_type"}
        ]
        for column in numeric_columns:
            if column in {"game_date", "season"}:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = _compute_rest_features(frame)
        frame = _compute_starter_flag(frame)
        frame = frame.merge(player_context, on="player_id", how="left")
        frame = _merge_team_context(frame, team_context)
        season_frames.append(frame)

    combined = pd.concat(season_frames, ignore_index=True)
    combined["line_points"] = pd.NA
    combined["line_assists"] = pd.NA
    combined["line_rebounds"] = pd.NA
    combined["closing_over_price"] = pd.NA
    combined["closing_under_price"] = pd.NA
    return _ensure_final_schema(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a player-game CSV dataset from nba_api.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help='One or more NBA seasons in YYYY-YY format, for example "2024-25".',
    )
    parser.add_argument("--season-type", default="Regular Season", help='Season type, for example "Regular Season".')
    parser.add_argument("--output", default="data/player_game_logs.csv", help="Path to write the CSV dataset.")
    args = parser.parse_args()

    dataset = build_dataset(args.seasons, args.season_type)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Wrote {len(dataset):,} rows to {output_path}")


if __name__ == "__main__":
    main()
