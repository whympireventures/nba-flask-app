from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from ncaa_api_client import NCAAApiClient, default_ncaa_season_year
from ncaa_features import build_feature_row


MODEL_FILE_DEFAULTS = {
    "points": "ncaa_model_points.pkl",
    "rebounds": "ncaa_model_rebounds.pkl",
    "assists": "ncaa_model_assists.pkl",
    "minutes": "ncaa_model_minutes.pkl",
    "fg_pct": "ncaa_model_fg_pct.pkl",
    "three_pt_pct": "ncaa_model_three_pt_pct.pkl",
    "turnovers": "ncaa_model_turnovers.pkl",
}

STAT_ALIASES = {
    "points": "points",
    "point": "points",
    "pts": "points",
    "rebounds": "rebounds",
    "rebound": "rebounds",
    "reb": "rebounds",
    "rebs": "rebounds",
    "assists": "assists",
    "assist": "assists",
    "ast": "assists",
    "asts": "assists",
    "minutes": "minutes",
    "minute": "minutes",
    "min": "minutes",
    "fg%": "fg_pct",
    "fgpct": "fg_pct",
    "fg_pct": "fg_pct",
    "fgpercentage": "fg_pct",
    "fieldgoalpct": "fg_pct",
    "field_goal_pct": "fg_pct",
    "fieldgoalpercentage": "fg_pct",
    "3p%": "three_pt_pct",
    "3pt%": "three_pt_pct",
    "3p_pct": "three_pt_pct",
    "3pt_pct": "three_pt_pct",
    "3ppct": "three_pt_pct",
    "3ptpct": "three_pt_pct",
    "threeptpct": "three_pt_pct",
    "three_pt_pct": "three_pt_pct",
    "threepointpercentage": "three_pt_pct",
    "three_pt_percentage": "three_pt_pct",
    "turnovers": "turnovers",
    "turnover": "turnovers",
    "to": "turnovers",
    "tov": "turnovers",
}

ERROR_BAND_FALLBACKS = {
    "points": 4.0,
    "rebounds": 2.5,
    "assists": 2.0,
    "minutes": 3.5,
    "fg_pct": 6.0,
    "three_pt_pct": 8.0,
    "turnovers": 1.5,
}

ROUNDING = {
    "points": 1,
    "rebounds": 1,
    "assists": 1,
    "minutes": 1,
    "fg_pct": 1,
    "three_pt_pct": 1,
    "turnovers": 1,
}


@dataclass(frozen=True)
class ModelSpec:
    target: str
    artifact_name: str
    feature_names: list[str] | None = None


@dataclass(frozen=True)
class PredictorBundle:
    specs: dict[str, ModelSpec]
    models: dict[str, Any]
    metadata: dict[str, Any]

    def _feature_frame_for_target(self, feature_row: dict[str, float], target: str) -> pd.DataFrame:
        model = self.models[target]
        spec = self.specs[target]
        columns = spec.feature_names
        if not columns:
            model_columns = getattr(model, "feature_names_in_", None)
            columns = list(model_columns) if model_columns is not None else None
        if columns:
            return pd.DataFrame([feature_row]).reindex(columns=columns, fill_value=0.0)
        return pd.DataFrame([feature_row])

    def predict(self, game_logs: list[dict[str, Any]], upcoming_context: dict[str, Any] | None = None) -> dict[str, Any]:
        feature_row = build_feature_row(game_logs, upcoming_context=upcoming_context)
        projections = _heuristic_projection_map(feature_row)
        projection_sources = {target: "heuristic" for target in projections}

        for target in self.models:
            frame = self._feature_frame_for_target(feature_row, target)
            result = self.models[target].predict(frame)
            projections[target] = float(result[0])
            projection_sources[target] = "model"

        expected_minutes = projections.get("minutes", feature_row.get("projected_minutes", 0.0))
        projections["expected_minutes"] = float(expected_minutes)
        return {
            "feature_row": feature_row,
            "projections": projections,
            "projection_sources": projection_sources,
        }

    def error_band_for_target(self, target: str) -> float:
        metrics = self.metadata.get("targets", {}).get(target, {}).get("metrics", {})
        return float(metrics.get("mae", ERROR_BAND_FALLBACKS.get(target, 3.0)) or ERROR_BAND_FALLBACKS.get(target, 3.0))


def _normalize_stat_type(value: str) -> str:
    normalized = "".join(character for character in str(value or "").strip().lower() if character.isalnum() or character in {"_", "%"})
    target = STAT_ALIASES.get(normalized)
    if not target:
        supported = ", ".join(sorted({alias for alias in STAT_ALIASES.values()}))
        raise ValueError(f"Unsupported NCAA stat type '{value}'. Supported stats: {supported}.")
    return target


def _round_stat(target: str, value: float) -> float:
    return round(float(value), ROUNDING.get(target, 1))


def _blend(base: float, alternate: float, weight: float) -> float:
    return (base * (1.0 - weight)) + (alternate * weight)


def _heuristic_projection(feature_row: dict[str, float], target: str) -> float:
    rolling_3 = float(feature_row.get(f"{target}_rolling_3", 0.0) or 0.0)
    rolling_5 = float(feature_row.get(f"{target}_rolling_5", 0.0) or 0.0)
    rolling_10 = float(feature_row.get(f"{target}_rolling_10", 0.0) or 0.0)
    season_avg = float(feature_row.get(f"{target}_season_avg", 0.0) or 0.0)
    opponent_avg = float(feature_row.get(f"vs_opp_{target}_avg", season_avg) or season_avg)
    opponent_games = float(feature_row.get("vs_opp_games_count", 0.0) or 0.0)

    projection = (rolling_3 * 0.5) + (rolling_5 * 0.3) + (season_avg * 0.2)
    if rolling_10:
        projection += (rolling_3 - rolling_10) * 0.1

    if opponent_games > 0:
        projection = _blend(projection, opponent_avg, 0.2)

    if target in {"points", "rebounds", "assists", "turnovers"}:
        projected_minutes = float(feature_row.get("projected_minutes", rolling_5) or rolling_5)
        baseline_minutes = max(float(feature_row.get("minutes_season_avg", projected_minutes) or projected_minutes), 1.0)
        minutes_scale = max(0.75, min(1.25, projected_minutes / baseline_minutes))
        projection *= minutes_scale
        projection = max(0.0, projection)
    elif target in {"fg_pct", "three_pt_pct"}:
        projection = max(0.0, min(100.0, projection))
    else:
        projection = max(0.0, projection)

    return projection


def _heuristic_projection_map(feature_row: dict[str, float]) -> dict[str, float]:
    projections = {
        target: _heuristic_projection(feature_row, target)
        for target in MODEL_FILE_DEFAULTS
    }
    projections["expected_minutes"] = projections["minutes"]
    return projections


def _load_metadata(model_dir: Path) -> dict[str, Any] | None:
    metadata_path = model_dir / "ncaa_model_metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


@lru_cache(maxsize=1)
def load_predictor_bundle() -> PredictorBundle:
    model_dir = settings.model_dir
    metadata = _load_metadata(model_dir) or {}

    specs: dict[str, ModelSpec] = {}
    if metadata.get("targets"):
        for target, payload in metadata["targets"].items():
            specs[target] = ModelSpec(
                target=target,
                artifact_name=payload["artifact_name"],
                feature_names=payload.get("feature_names"),
            )
    else:
        for target, artifact_name in MODEL_FILE_DEFAULTS.items():
            artifact_path = model_dir / artifact_name
            if artifact_path.exists():
                specs[target] = ModelSpec(target=target, artifact_name=artifact_name)

    models: dict[str, Any] = {}
    for target, spec in specs.items():
        artifact_path = model_dir / spec.artifact_name
        if not artifact_path.exists():
            continue
        with open(artifact_path, "rb") as artifact:
            models[target] = pickle.load(artifact)

    specs = {target: spec for target, spec in specs.items() if target in models}
    return PredictorBundle(specs=specs, models=models, metadata=metadata)


def predict_player_statline(
    player_name: str,
    *,
    opponent: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    client = NCAAApiClient()
    player = client.resolve_player_by_name(player_name)
    if player is None:
        raise ValueError(f"No NCAA player match found for '{player_name}'.")

    season_value = int(season if season is not None else default_ncaa_season_year())
    game_logs = client.get_player_statistics(
        str(player["id"]),
        season=season_value,
        team_id=player.get("team_id"),
    )
    if not game_logs:
        raise ValueError(f"No NCAA game logs found for '{player['display_name']}' in season {season_value}.")

    upcoming_context = {"opponent": opponent} if opponent else None
    bundle = load_predictor_bundle()
    prediction_payload = bundle.predict(game_logs, upcoming_context=upcoming_context)
    projections = prediction_payload["projections"]

    return {
        "player_id": str(player["id"]),
        "player_name": player["display_name"],
        "team_id": str(player.get("team_id") or ""),
        "team_name": str(player.get("team_name") or ""),
        "season": season_value,
        "opponent": opponent or "",
        "points": _round_stat("points", projections["points"]),
        "rebounds": _round_stat("rebounds", projections["rebounds"]),
        "assists": _round_stat("assists", projections["assists"]),
        "minutes": _round_stat("minutes", projections["minutes"]),
        "fg_pct": _round_stat("fg_pct", projections["fg_pct"]),
        "three_pt_pct": _round_stat("three_pt_pct", projections["three_pt_pct"]),
        "turnovers": _round_stat("turnovers", projections["turnovers"]),
        "expected_minutes": _round_stat("minutes", projections["expected_minutes"]),
        "projection_sources": prediction_payload["projection_sources"],
        "model_source": "models" if any(source == "model" for source in prediction_payload["projection_sources"].values()) else "heuristic",
        "feature_row": prediction_payload["feature_row"],
        "error_bands": {
            target: bundle.error_band_for_target(target)
            for target in MODEL_FILE_DEFAULTS
        },
    }


def predict_player_prop(
    player_name: str,
    *,
    stat_type: str,
    stat_line: float,
    opponent: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    normalized_stat = _normalize_stat_type(stat_type)
    prediction_payload = predict_player_statline(
        player_name,
        opponent=opponent,
        season=season,
    )
    projected_stat = float(prediction_payload[normalized_stat])
    line_value = float(stat_line)
    edge = projected_stat - line_value
    error_band = float(prediction_payload["error_bands"].get(normalized_stat, ERROR_BAND_FALLBACKS[normalized_stat]))
    ratio = abs(edge) / max(error_band, 0.5)

    if ratio >= 1.2:
        confidence_label = "High"
    elif ratio >= 0.7:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return {
        "player_id": prediction_payload["player_id"],
        "player_name": prediction_payload["player_name"],
        "team_id": prediction_payload["team_id"],
        "team_name": prediction_payload["team_name"],
        "season": prediction_payload["season"],
        "opponent": prediction_payload["opponent"],
        "stat_type": normalized_stat,
        "stat_line": _round_stat(normalized_stat, line_value),
        "projected_stat": _round_stat(normalized_stat, projected_stat),
        "pick": "over" if edge >= 0 else "under",
        "edge": _round_stat(normalized_stat, edge),
        "confidence": {
            "label": confidence_label,
            "edge_to_error_ratio": round(ratio, 2),
            "error_band": round(error_band, 2),
        },
        "projection_source": prediction_payload["projection_sources"].get(normalized_stat, "heuristic"),
        "all_projections": {
            "points": prediction_payload["points"],
            "rebounds": prediction_payload["rebounds"],
            "assists": prediction_payload["assists"],
            "minutes": prediction_payload["minutes"],
            "fg_pct": prediction_payload["fg_pct"],
            "three_pt_pct": prediction_payload["three_pt_pct"],
            "turnovers": prediction_payload["turnovers"],
        },
        "model_source": prediction_payload["model_source"],
    }
