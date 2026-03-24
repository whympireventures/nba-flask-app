from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import settings
from modeling import load_predictor_bundle


GROUP_PATTERNS = {
    "Recent Form": [
        "points_rolling",
        "assists_rolling",
        "rebounds_rolling",
        "_season_avg",
        "_last_game",
        "last_game_",
        "usage_trend",
    ],
    "Minutes And Role": [
        "minutes_",
        "projected_minutes",
        "projected_opportunity",
        "lead_role",
        "heavy_load",
        "rotation_role",
        "bench_risk",
        "home_minutes_interaction",
    ],
    "Matchup And Team Context": [
        "team_",
        "opp_",
        "pace_edge",
        "defense_edge",
        "win_pct_edge",
        "home",
        "rest_days",
        "is_back_to_back",
        "rest_advantage",
        "back_to_back_usage_penalty",
    ],
    "Efficiency And Usage": [
        "fg_pct",
        "three_pt_pct",
        "ft_pct",
        "usage_rate",
        "true_shooting_pct",
        "player_pace",
        "minutes_usage_interaction",
    ],
    "Sportsbook Inputs": [
        "line_",
        "closing_",
    ],
}


def _load_metadata() -> dict[str, Any]:
    metadata_path = Path(settings.model_dir) / "model_metadata.json"
    if not metadata_path.exists():
        return {"targets": {}}
    return json.loads(metadata_path.read_text())


def _group_name(feature_name: str) -> str:
    for group_name, patterns in GROUP_PATTERNS.items():
        if any(pattern in feature_name for pattern in patterns):
            return group_name
    return "Other"


@lru_cache(maxsize=1)
def get_model_insights() -> dict[str, Any]:
    bundle = load_predictor_bundle()
    metadata = _load_metadata()
    targets: list[dict[str, Any]] = []

    for target_name in ("points", "assists", "rebounds"):
        spec = bundle.specs[target_name]
        model = bundle.models[target_name]
        regressor = model.named_steps["regressor"] if hasattr(model, "named_steps") else model
        feature_names = spec.feature_names or []
        raw_importances = getattr(regressor, "feature_importances_", [])
        importances = [float(value) for value in raw_importances]
        total_importance = sum(importances) or 1.0

        ranked_features = sorted(
            [
                {
                    "name": feature_name,
                    "importance": importance,
                    "share_pct": round((importance / total_importance) * 100, 1),
                    "group": _group_name(feature_name),
                }
                for feature_name, importance in zip(feature_names, importances)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )

        grouped_totals: dict[str, float] = {}
        for row in ranked_features:
            grouped_totals[row["group"]] = grouped_totals.get(row["group"], 0.0) + row["importance"]

        groups = sorted(
            [
                {
                    "name": group_name,
                    "importance": importance,
                    "share_pct": round((importance / total_importance) * 100, 1),
                }
                for group_name, importance in grouped_totals.items()
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )

        metrics = metadata.get("targets", {}).get(target_name, {}).get("metrics", {})
        targets.append(
            {
                "name": target_name.title(),
                "metrics": {
                    "mae": round(float(metrics.get("mae", 0.0)), 2),
                    "r2": round(float(metrics.get("r2", 0.0)), 2),
                },
                "top_features": ranked_features[:15],
                "top_groups": groups,
            }
        )

    return {"targets": targets}
