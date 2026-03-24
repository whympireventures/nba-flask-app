from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from features import build_feature_row, build_legacy_feature_frame


@dataclass(frozen=True)
class ModelSpec:
    target: str
    artifact_name: str
    feature_names: list[str] | None = None


@dataclass(frozen=True)
class PredictorBundle:
    specs: dict[str, ModelSpec]
    models: dict[str, Any]
    auxiliary_specs: dict[str, ModelSpec]
    auxiliary_models: dict[str, Any]
    metadata: dict[str, Any]

    def _error_band_for_target(self, target: str) -> float:
        metrics = self.metadata.get("targets", {}).get(target, {}).get("metrics", {})
        fallback = {"points": 4.5, "assists": 1.8, "rebounds": 2.2}.get(target, 3.0)
        return float(metrics.get("mae", fallback) or fallback)

    def _baseline_confidence_for_target(self, target: str) -> float:
        metrics = self.metadata.get("targets", {}).get(target, {}).get("metrics", {})
        key = "within_5" if target == "points" else "within_2"
        fallback = {"points": 0.66, "assists": 0.8, "rebounds": 0.63}.get(target, 0.6)
        return float(metrics.get(key, fallback) or fallback)

    def _build_confidence_summary(
        self,
        rich_feature_row: dict[str, float],
        *,
        expected_minutes: float,
    ) -> dict[str, dict[str, float | str]]:
        summary: dict[str, dict[str, float | str]] = {}
        thresholds = {"points": 5.0, "assists": 2.0, "rebounds": 2.0}

        for target in ("points", "assists", "rebounds"):
            baseline = self._baseline_confidence_for_target(target)
            rolling_std = float(rich_feature_row.get(f"{target}_rolling_std_10", 0.0) or 0.0)
            bench_risk = float(rich_feature_row.get("bench_risk_rate_5", 0.0) or 0.0)
            lead_role = float(rich_feature_row.get("lead_role_rate_5", 0.0) or 0.0)
            is_back_to_back = float(rich_feature_row.get("is_back_to_back", 0.0) or 0.0)
            minutes_trend = abs(float(rich_feature_row.get("minutes_trend", 0.0) or 0.0))

            volatility_penalty = min(0.18, (rolling_std / max(thresholds[target], 1.0)) * 0.12)
            low_minutes_penalty = min(0.16, max(0.0, 28.0 - expected_minutes) * 0.01)
            bench_penalty = min(0.1, bench_risk * 0.1)
            back_to_back_penalty = 0.03 if is_back_to_back else 0.0
            trend_penalty = min(0.05, minutes_trend * 0.008)
            lead_role_bonus = min(0.05, lead_role * 0.05)

            score = baseline - volatility_penalty - low_minutes_penalty - bench_penalty - back_to_back_penalty - trend_penalty + lead_role_bonus
            score = max(0.2, min(0.92, score))
            if score >= 0.75:
                label = "High"
            elif score >= 0.55:
                label = "Medium"
            else:
                label = "Low"

            summary[target] = {
                "score": round(score * 100, 1),
                "label": label,
                "error_band": round(self._error_band_for_target(target), 1),
            }
        return summary

    def predict(self, game_logs: list[dict[str, Any]], upcoming_context: dict[str, Any] | None = None) -> dict[str, float]:
        rich_feature_row = build_feature_row(game_logs, upcoming_context=upcoming_context)
        rich_frame = pd.DataFrame([rich_feature_row])
        legacy_frame = build_legacy_feature_frame(game_logs)
        predictions: dict[str, float] = {}

        for target, spec in self.specs.items():
            model = self.models[target]
            if spec.feature_names:
                frame = rich_frame.reindex(columns=spec.feature_names, fill_value=0.0)
            else:
                frame = legacy_frame
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
                result = model.predict(frame)
            predictions[target] = float(result[0])

        expected_minutes = float(rich_feature_row.get("projected_minutes", 0.0) or 0.0)
        minutes_spec = self.auxiliary_specs.get("minutes")
        minutes_model = self.auxiliary_models.get("minutes")
        if minutes_spec and minutes_model:
            minutes_frame = rich_frame.reindex(columns=minutes_spec.feature_names or [], fill_value=0.0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
                model_minutes = float(minutes_model.predict(minutes_frame)[0])
            heuristic_minutes = float(rich_feature_row.get("projected_minutes", model_minutes) or model_minutes)
            expected_minutes = (model_minutes * 0.7) + (heuristic_minutes * 0.3)
        expected_minutes = max(8.0, min(42.0, expected_minutes))
        predictions["expected_minutes"] = round(expected_minutes, 1)
        predictions["minutes_baseline"] = round(float(rich_feature_row.get("projected_minutes", 0.0) or 0.0), 1)
        predictions["confidence_summary"] = self._build_confidence_summary(
            rich_feature_row,
            expected_minutes=expected_minutes,
        )

        return predictions


def _load_metadata(model_dir: Path) -> dict[str, Any] | None:
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


@lru_cache(maxsize=1)
def load_predictor_bundle() -> PredictorBundle:
    model_dir = settings.model_dir
    metadata = _load_metadata(model_dir) or {}

    if metadata:
        specs = {
            target: ModelSpec(
                target=target,
                artifact_name=payload["artifact_name"],
                feature_names=payload.get("feature_names"),
            )
            for target, payload in metadata["targets"].items()
        }
    else:
        specs = {
            "points": ModelSpec(target="points", artifact_name="model_points.pkl"),
            "assists": ModelSpec(target="assists", artifact_name="model_assists.pkl"),
            "rebounds": ModelSpec(target="rebounds", artifact_name="model_rebounds.pkl"),
        }

    models = {}
    for target, spec in specs.items():
        artifact_path = model_dir / spec.artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        with open(artifact_path, "rb") as artifact:
            models[target] = pickle.load(artifact)

    auxiliary_specs = {
        target: ModelSpec(
            target=target,
            artifact_name=payload["artifact_name"],
            feature_names=payload.get("feature_names"),
        )
        for target, payload in metadata.get("auxiliary", {}).items()
    }
    auxiliary_models = {}
    for target, spec in auxiliary_specs.items():
        artifact_path = model_dir / spec.artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Auxiliary model artifact not found: {artifact_path}")
        with open(artifact_path, "rb") as artifact:
            auxiliary_models[target] = pickle.load(artifact)

    return PredictorBundle(
        specs=specs,
        models=models,
        auxiliary_specs=auxiliary_specs,
        auxiliary_models=auxiliary_models,
        metadata=metadata,
    )
