import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread

from flask import Flask, Response, render_template, request

from api_client import NBAApiClient
from config import settings
from historical_backtest import get_historical_backtest_overview, run_historical_backtest, run_batch_backtest
try:
    from model_insights import get_model_insights
    _MODEL_INSIGHTS_AVAILABLE = True
except ImportError:
    _MODEL_INSIGHTS_AVAILABLE = False
    get_model_insights = None

from ncaa_prediction import predict_player_prop as predict_ncaa_player_prop

try:
    from odds_api_client import OddsApiClient, OddsApiProviderError
    _ODDS_API_AVAILABLE = True
except ImportError:
    _ODDS_API_AVAILABLE = False
    OddsApiClient = None
    OddsApiProviderError = Exception
from parlayplay_client import ParlayPlayClient, ParlayPlayProviderError
from prediction import predict_player_statline, predict_player_stats
from prizepicks_client import PrizePicksClient, PrizePicksProviderError
from underdog_client import UnderdogClient, UnderdogProviderError

app = Flask(__name__)
client = NBAApiClient()
prizepicks_client = PrizePicksClient()
parlayplay_client = ParlayPlayClient()
underdog_client = UnderdogClient()
odds_api_client = OddsApiClient() if _ODDS_API_AVAILABLE else None
_underdog_prewarm_lock = Lock()
_underdog_prewarm_started = False
NBA_TEAM_OPTIONS = [
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]
TRACKED_MARKETS = [
    ("Points", "line_points"),
    ("Assists", "line_assists"),
    ("Rebounds", "line_rebounds"),
    ("Points + Rebounds", "line_points_rebounds"),
    ("Points + Assists", "line_points_assists"),
    ("Assists + Rebounds", "line_assists_rebounds"),
    ("PRA", "line_points_rebounds_assists"),
]
UNDERDOG_MARKET_FILTERS = ["all"] + [market_label for market_label, _ in TRACKED_MARKETS]
UNDERDOG_BOARD_PREDICTION_WORKERS = 6
TRACKING_FIELDNAMES = [
    "created_at",
    "sportsbook",
    "player_id",
    "player_name",
    "opponent_abbr",
    "game_date",
    "market",
    "model_projection",
    "sportsbook_line",
    "pick_side",
    "payout_multiplier",
    "edge",
    "edge_label",
    "data_mode",
    "actual_points",
    "actual_assists",
    "actual_rebounds",
    "actual_result",
    "prediction_error",
    "absolute_error",
    "within_tolerance",
    "pick_result",
    "pick_hit",
    "scored_at",
]
EDGE_SIGNAL_THRESHOLDS = {
    "Points": {"lean": 3.0, "best": 4.0},
    "Assists": {"lean": 1.5, "best": 2.0},
    "Rebounds": {"lean": 1.5, "best": 2.0},
    "Points + Rebounds": {"lean": 3.0, "best": 4.0},
    "Points + Assists": {"lean": 3.0, "best": 4.0},
    "Assists + Rebounds": {"lean": 2.0, "best": 3.0},
    "PRA": {"lean": 4.0, "best": 5.0},
}
BOARD_PLAYERS_PER_PAGE = 50
ACTUAL_IMPORT_ALIASES = {
    "player_name": ["player_name", "name", "player"],
    "player_id": ["player_id", "id"],
    "game_date": ["game_date", "date"],
    "opponent_abbr": ["opponent_abbr", "opponent", "opp", "team"],
    "actual_points": ["actual_points", "points", "pts"],
    "actual_assists": ["actual_assists", "assists", "ast"],
    "actual_rebounds": ["actual_rebounds", "rebounds", "rebs", "reb"],
}
MARKET_TOLERANCE = {
    "Points": 4.0,
    "Assists": 2.0,
    "Rebounds": 2.0,
    "Points + Rebounds": 4.0,
    "Points + Assists": 4.0,
    "Assists + Rebounds": 3.0,
    "PRA": 5.0,
}
# Historical pick hit rates per market (from prediction_tracking.csv)
MARKET_HIT_RATE = {
    "Points": 0.624,
    "Rebounds": 0.700,
    "Assists": 0.843,
    "Points + Rebounds": 0.530,
    "Points + Assists": 0.569,
    "Assists + Rebounds": 0.727,
    "PRA": 0.583,
}
IMPORT_FIELD_ALIASES = {
    "player_name": ["player_name", "name", "player"],
    "player_id": ["player_id", "id"],
    "sportsbook": ["sportsbook", "book", "sports_book", "platform"],
    "team_abbr": ["team_abbr", "team"],
    "opponent_abbr": ["opponent_abbr", "opponent", "opp"],
    "game_date": ["game_date", "date"],
    "market": ["market", "stat", "prop", "wager_type"],
    "line": ["line", "value", "projection", "prop_line"],
    "pick_side": ["pick_side", "side", "selection", "pick", "direction"],
    "payout_multiplier": ["payout_multiplier", "multiplier", "payout", "odds", "price"],
    "line_points": ["line_points", "points", "pts"],
    "line_assists": ["line_assists", "assists", "ast"],
    "line_rebounds": ["line_rebounds", "rebounds", "reb", "rebs"],
    "line_points_rebounds": ["line_points_rebounds", "points_rebounds", "pts_rebs", "pr"],
    "line_points_assists": ["line_points_assists", "points_assists", "pts_asts", "pa"],
    "line_assists_rebounds": ["line_assists_rebounds", "line_rebounds_assists", "assists_rebounds", "rebounds_assists", "asts_rebs", "rebs_asts", "ar"],
    "line_points_rebounds_assists": ["line_points_rebounds_assists", "points_rebounds_assists", "pra"],
}
RAW_MARKET_TO_KEY = {
    "points": "line_points",
    "assists": "line_assists",
    "rebounds": "line_rebounds",
    "pra": "line_points_rebounds_assists",
    "rebs+asts": "line_assists_rebounds",
    "asts+rebs": "line_assists_rebounds",
    "reb+ast": "line_assists_rebounds",
    "ast+reb": "line_assists_rebounds",
    "rebounds+assists": "line_assists_rebounds",
    "assists+rebounds": "line_assists_rebounds",
    "pts+rebs": "line_points_rebounds",
    "pts+asts": "line_points_assists",
    "pts+reb": "line_points_rebounds",
    "pts+ast": "line_points_assists",
    "points+rebounds": "line_points_rebounds",
    "points+assists": "line_points_assists",
}


def current_data_mode() -> dict[str, str]:
    if settings.rapidapi_key:
        return {
            "label": "RapidAPI Mode",
            "detail": "Paid live API connected",
        }
    return {
        "label": "Free Data Mode",
        "detail": "Using nba_api fallback",
    }


def _parse_start_date(raw_value: str | None) -> str:
    if not raw_value:
        return ""
    normalized = str(raw_value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        return ""


def _build_prediction_summary(points: float, assists: float, rebounds: float) -> dict[str, float]:
    return {
        "Points": round(points, 1),
        "Assists": round(assists, 1),
        "Rebounds": round(rebounds, 1),
        "Points + Rebounds": round(points + rebounds, 1),
        "Points + Assists": round(points + assists, 1),
        "Assists + Rebounds": round(assists + rebounds, 1),
        "PRA": round(points + assists + rebounds, 1),
    }


def _parse_optional_float(raw_value: str | None) -> float | None:
    if raw_value in (None, ""):
        return None
    try:
        return float(str(raw_value).strip())
    except ValueError:
        return None


def _parse_optional_int(raw_value: str | None) -> int | None:
    if raw_value in (None, ""):
        return None
    try:
        return int(str(raw_value).strip())
    except ValueError:
        return None


def _normalize_pick_side(raw_value: str | None) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"more", "over", "higher", "up"}:
        return "more"
    if normalized in {"less", "under", "lower", "down"}:
        return "less"
    return ""


def _pick_side_label(pick_side: str | None) -> str:
    normalized = _normalize_pick_side(pick_side)
    if normalized == "more":
        return "More"
    if normalized == "less":
        return "Less"
    return ""


def _normalize_tracking_row(row: dict[str, str]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in TRACKING_FIELDNAMES}


def _ensure_tracking_file_schema() -> list[dict[str, str]]:
    tracking_path = settings.tracking_file
    if not tracking_path.exists():
        return []

    with tracking_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [_normalize_tracking_row(row) for row in reader]
        existing_fieldnames = reader.fieldnames or []

    if existing_fieldnames == TRACKING_FIELDNAMES:
        return rows

    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    with tracking_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRACKING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _normalize_actual_field(field_name: str) -> str:
    normalized = str(field_name or "").strip().lower()
    for canonical, aliases in ACTUAL_IMPORT_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def _market_actual_value(market_label: str, *, points: float, assists: float, rebounds: float) -> float:
    if market_label == "Points":
        return points
    if market_label == "Assists":
        return assists
    if market_label == "Rebounds":
        return rebounds
    if market_label == "Points + Rebounds":
        return points + rebounds
    if market_label == "Points + Assists":
        return points + assists
    if market_label == "Assists + Rebounds":
        return assists + rebounds
    if market_label == "PRA":
        return points + assists + rebounds
    return 0.0


def _accuracy_tier(mean_ratio: float) -> str:
    if mean_ratio <= 1.0:
        return "Elite"
    if mean_ratio <= 1.15:
        return "Strong"
    if mean_ratio <= 1.35:
        return "Solid"
    if mean_ratio <= 1.6:
        return "Needs Work"
    return "Rebuild"


def _confidence_label(within_rate: float | None, sample_count: int) -> str:
    if sample_count < 3 or within_rate is None:
        return "Provisional"
    if within_rate >= 75:
        return "High"
    if within_rate >= 55:
        return "Medium"
    return "Low"


def _label_from_confidence_score(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"


def _build_market_confidence_summary(prediction_payload: dict[str, object]) -> dict[str, dict[str, float | str]]:
    singles = prediction_payload.get("confidence_summary")
    if not isinstance(singles, dict):
        singles = {}

    summary: dict[str, dict[str, float | str]] = {}
    for market_label, target_key in (("Points", "points"), ("Assists", "assists"), ("Rebounds", "rebounds")):
        target_summary = singles.get(target_key, {})
        if not isinstance(target_summary, dict):
            target_summary = {}
        score = float(target_summary.get("score", 55.0) or 55.0)
        error_band = float(target_summary.get("error_band", 3.0) or 3.0)
        summary[market_label] = {
            "score": round(score, 1),
            "label": str(target_summary.get("label") or _label_from_confidence_score(score)),
            "error_band": round(error_band, 1),
        }

    combo_map = {
        "Points + Rebounds": ("Points", "Rebounds"),
        "Points + Assists": ("Points", "Assists"),
        "Assists + Rebounds": ("Assists", "Rebounds"),
        "PRA": ("Points", "Assists", "Rebounds"),
    }
    for market_label, components in combo_map.items():
        component_scores = [float(summary[name]["score"]) for name in components if name in summary]
        component_bands = [float(summary[name]["error_band"]) for name in components if name in summary]
        if not component_scores:
            continue
        score = max(25.0, min(90.0, sum(component_scores) / len(component_scores) - ((len(components) - 1) * 6.0)))
        error_band = sum(component_bands)
        summary[market_label] = {
            "score": round(score, 1),
            "label": _label_from_confidence_score(score),
            "error_band": round(error_band, 1),
        }

    return summary


def _extract_line_inputs(values) -> tuple[str, dict[str, float | None]]:
    sportsbook = values.get("sportsbook", "").strip() or "Manual"
    lines = {
        "line_points": _parse_optional_float(values.get("line_points")),
        "line_assists": _parse_optional_float(values.get("line_assists")),
        "line_rebounds": _parse_optional_float(values.get("line_rebounds")),
        "line_points_rebounds": _parse_optional_float(values.get("line_points_rebounds")),
        "line_points_assists": _parse_optional_float(values.get("line_points_assists")),
        "line_assists_rebounds": _parse_optional_float(values.get("line_assists_rebounds")),
        "line_points_rebounds_assists": _parse_optional_float(values.get("line_points_rebounds_assists")),
    }
    return sportsbook, lines


def _normalize_import_field(field_name: str) -> str:
    normalized = str(field_name or "").strip().lower()
    for canonical, aliases in IMPORT_FIELD_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def _coerce_import_row(
    raw_row: dict[str, str],
    *,
    default_sportsbook: str,
    default_game_date: str,
    default_opponent_abbr: str,
) -> dict[str, str | float | None]:
    normalized_row = {_normalize_import_field(key): value for key, value in raw_row.items()}
    sportsbook = str(normalized_row.get("sportsbook") or default_sportsbook or "Manual Import").strip()
    opponent_abbr = str(normalized_row.get("opponent_abbr") or default_opponent_abbr or "").strip().upper()
    team_abbr = str(normalized_row.get("team_abbr") or "").strip().upper()
    game_date = str(normalized_row.get("game_date") or default_game_date or "").strip()
    row = {
        "player_name": str(normalized_row.get("player_name") or "").strip(),
        "player_id": str(normalized_row.get("player_id") or "").strip(),
        "sportsbook": sportsbook,
        "team_abbr": team_abbr,
        "opponent_abbr": opponent_abbr,
        "game_date": game_date,
        "pick_side": _normalize_pick_side(str(normalized_row.get("pick_side") or "")),
        "payout_multiplier": _parse_optional_float(normalized_row.get("payout_multiplier")),
        "line_points": _parse_optional_float(normalized_row.get("line_points")),
        "line_assists": _parse_optional_float(normalized_row.get("line_assists")),
        "line_rebounds": _parse_optional_float(normalized_row.get("line_rebounds")),
        "line_points_rebounds": _parse_optional_float(normalized_row.get("line_points_rebounds")),
        "line_points_assists": _parse_optional_float(normalized_row.get("line_points_assists")),
        "line_assists_rebounds": _parse_optional_float(normalized_row.get("line_assists_rebounds")),
        "line_points_rebounds_assists": _parse_optional_float(normalized_row.get("line_points_rebounds_assists")),
    }
    market_key = RAW_MARKET_TO_KEY.get(_normalize_raw_market(normalized_row.get("market")))
    row["market_key"] = market_key or ""
    line_value = _parse_optional_float(normalized_row.get("line"))
    if market_key and line_value is not None and row.get(market_key) is None:
        row[market_key] = line_value
    return row


def _read_import_text() -> str:
    pasted_text = request.form.get("csv_text", "").strip()
    if pasted_text:
        return pasted_text
    uploaded_file = request.files.get("csv_file")
    if uploaded_file and uploaded_file.filename:
        return uploaded_file.read().decode("utf-8-sig")
    return ""


def _extract_opponent_abbr(raw_value: str, default_opponent_abbr: str) -> str:
    text = str(raw_value or "").strip().upper()
    if not text:
        return default_opponent_abbr
    match = re.search(r"([A-Z]{2,4})\s*$", text)
    if match:
        return match.group(1)
    return default_opponent_abbr


def _infer_import_opponent_abbr(
    *,
    player_id: str,
    player_name: str,
    team_abbr: str,
    game_date: str,
) -> str:
    if not game_date:
        return ""

    candidate_years = []
    try:
        parsed_date = datetime.fromisoformat(game_date).date()
        candidate_years = [parsed_date.year, parsed_date.year - 1]
    except ValueError:
        candidate_years = [settings.season_start_year, settings.season_start_year - 1]

    for season_start_year in candidate_years:
        try:
            game_logs = client.get_player_statistics(player_id, season_start_year=season_start_year)
        except Exception:
            continue
        for game in game_logs:
            raw_date = (game.get("game") or {}).get("date") if isinstance(game.get("game"), dict) else None
            normalized_date = _parse_start_date(raw_date) or str(raw_date or "")
            if normalized_date != game_date:
                continue
            game_team_abbr = str(((game.get("team") or {}).get("code")) or "").upper()
            if team_abbr and game_team_abbr and game_team_abbr != team_abbr:
                continue
            return str(game.get("opponent_abbr") or "").upper()
    return ""


def _normalize_raw_market(raw_value: str) -> str:
    return str(raw_value or "").strip().lower().replace(" ", "")


def _parse_raw_prizepicks_text(
    csv_text: str,
    *,
    default_sportsbook: str,
    default_game_date: str,
    default_opponent_abbr: str,
) -> tuple[list[dict[str, str | float | None]], list[str]]:
    errors: list[str] = []
    merged_rows: dict[tuple[str, str, str, str], dict[str, str | float | None]] = {}

    for index, raw_line in enumerate(csv_text.splitlines(), start=1):
        line = raw_line.strip().lstrip("\ufeff")
        if not line:
            continue
        parsed = next(csv.reader([line]), [])
        if not parsed:
            continue
        if parsed[0].strip().lower() == "player":
            continue
        if len(parsed) < 7:
            continue

        player_name = parsed[0].strip()
        if not player_name:
            continue

        market_key = RAW_MARKET_TO_KEY.get(_normalize_raw_market(parsed[-1]))
        if not market_key:
            continue

        line_value = _parse_optional_float(parsed[-2])
        if line_value is None:
            errors.append(f"Row {index}: could not read a line value for {player_name}.")
            continue

        opponent_abbr = _extract_opponent_abbr(parsed[3], default_opponent_abbr)
        sportsbook = default_sportsbook or "Manual Import"
        game_date = default_game_date
        merge_key = (player_name.lower(), opponent_abbr, game_date, sportsbook)
        row = merged_rows.setdefault(
            merge_key,
            {
                "player_name": player_name,
                "player_id": "",
                "sportsbook": sportsbook,
                "opponent_abbr": opponent_abbr,
                "game_date": game_date,
                "line_points": None,
                "line_assists": None,
                "line_rebounds": None,
                "line_points_rebounds": None,
                "line_points_assists": None,
                "line_assists_rebounds": None,
                "line_points_rebounds_assists": None,
            },
        )
        row[market_key] = line_value

    if not merged_rows and not errors:
        errors.append("The text file did not match the supported raw PrizePicks export format.")
    return list(merged_rows.values()), errors


def _parse_import_rows(
    csv_text: str,
    *,
    default_sportsbook: str,
    default_game_date: str,
    default_opponent_abbr: str,
) -> tuple[list[dict[str, str | float | None]], list[str]]:
    errors: list[str] = []
    if not csv_text.strip():
        return [], ["Paste CSV text or upload a CSV file first."]

    lines = csv_text.lstrip("\ufeff").splitlines()
    first_line = lines[0].strip().lower() if lines else ""
    if first_line.startswith("player,team,position,opponent,game time,line,market") or (
        "market" in csv_text.lower() and "player_name" not in first_line and "sportsbook" not in first_line
    ):
        return _parse_raw_prizepicks_text(
            csv_text,
            default_sportsbook=default_sportsbook,
            default_game_date=default_game_date,
            default_opponent_abbr=default_opponent_abbr,
        )

    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return [], ["The CSV needs a header row."]

    rows: list[dict[str, str | float | None]] = []
    for index, raw_row in enumerate(reader, start=2):
        if not raw_row or not any(str(value or "").strip() for value in raw_row.values()):
            continue
        row = _coerce_import_row(
            raw_row,
            default_sportsbook=default_sportsbook,
            default_game_date=default_game_date,
            default_opponent_abbr=default_opponent_abbr,
        )
        if not row["player_name"] and not row["player_id"]:
            errors.append(f"Row {index}: include either player_name or player_id.")
            continue
        if not any(
            row[key] is not None
            for key in (
                "line_points",
                "line_assists",
                "line_rebounds",
                "line_points_rebounds",
                "line_points_assists",
                "line_assists_rebounds",
                "line_points_rebounds_assists",
            )
        ):
            errors.append(f"Row {index}: add at least one line column.")
            continue
        if (row.get("pick_side") or row.get("payout_multiplier") is not None) and not (
            row.get("market_key")
            or sum(
                1
                for key in (
                    "line_points",
                    "line_assists",
                    "line_rebounds",
                    "line_points_rebounds",
                    "line_points_assists",
                    "line_assists_rebounds",
                    "line_points_rebounds_assists",
                )
                if row.get(key) is not None
            ) == 1
        ):
            errors.append(f"Row {index}: use pick_side and multiplier with a single market row (`market` + `line`).")
            continue
        rows.append(row)
    if not rows and not errors:
        errors.append("No usable rows were found in the CSV.")
    return rows, errors


def _parse_actual_rows(
    csv_text: str,
    *,
    default_game_date: str,
    default_opponent_abbr: str,
) -> tuple[list[dict[str, str | float]], list[str]]:
    errors: list[str] = []
    if not csv_text.strip():
        return [], ["Paste actual results CSV text or upload a file first."]

    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    if not reader.fieldnames:
        return [], ["The actual-results CSV needs a header row."]

    rows: list[dict[str, str | float]] = []
    for index, raw_row in enumerate(reader, start=2):
        if not raw_row or not any(str(value or "").strip() for value in raw_row.values()):
            continue
        normalized = {_normalize_actual_field(key): value for key, value in raw_row.items()}
        points = _parse_optional_float(normalized.get("actual_points"))
        assists = _parse_optional_float(normalized.get("actual_assists"))
        rebounds = _parse_optional_float(normalized.get("actual_rebounds"))
        if points is None or assists is None or rebounds is None:
            errors.append(f"Row {index}: include actual_points, actual_assists, and actual_rebounds.")
            continue
        player_name = str(normalized.get("player_name") or "").strip()
        player_id = str(normalized.get("player_id") or "").strip()
        if not player_name and not player_id:
            errors.append(f"Row {index}: include player_name or player_id.")
            continue
        rows.append(
            {
                "player_name": player_name,
                "player_id": player_id,
                "game_date": str(normalized.get("game_date") or default_game_date or "").strip(),
                "opponent_abbr": str(normalized.get("opponent_abbr") or default_opponent_abbr or "").strip().upper(),
                "actual_points": points,
                "actual_assists": assists,
                "actual_rebounds": rebounds,
            }
        )
    if not rows and not errors:
        errors.append("No usable actual-result rows were found in the CSV.")
    return rows, errors


def _normalize_player_lookup(value: str) -> str:
    return " ".join(str(value or "").lower().replace(".", "").split())


def _score_pick_result(edge_value: float | None, line_value: str, actual_result: float) -> tuple[str, str]:
    if edge_value is None or not line_value:
        return "", ""
    try:
        sportsbook_line = float(line_value)
    except ValueError:
        return "", ""
    if actual_result > sportsbook_line:
        actual_side = "Over"
    elif actual_result < sportsbook_line:
        actual_side = "Under"
    else:
        actual_side = "Push"
    if actual_side == "Push":
        return actual_side, ""
    if edge_value > 0:
        return actual_side, "1" if actual_side == "Over" else "0"
    if edge_value < 0:
        return actual_side, "1" if actual_side == "Under" else "0"
    return actual_side, ""


def _score_pick_result_for_side(
    *,
    edge_value: float | None,
    line_value: str,
    actual_result: float,
    pick_side: str = "",
) -> tuple[str, str]:
    normalized_side = _normalize_pick_side(pick_side)
    if not normalized_side:
        return _score_pick_result(edge_value, line_value, actual_result)
    try:
        sportsbook_line = float(line_value)
    except ValueError:
        return "", ""
    if actual_result > sportsbook_line:
        actual_side = "More"
    elif actual_result < sportsbook_line:
        actual_side = "Less"
    else:
        actual_side = "Push"
    if actual_side == "Push":
        return actual_side, ""
    expected_side = _pick_side_label(normalized_side)
    return actual_side, "1" if actual_side == expected_side else "0"


def _apply_actual_results(rows: list[dict[str, str | float]]) -> dict[str, int]:
    tracking_path = settings.tracking_file
    existing_rows = _ensure_tracking_file_schema()
    if not existing_rows:
        return {"matched_rows": 0, "updated_players": 0}

    matched_rows = 0
    updated_players = 0
    updated_indexes: set[int] = set()
    scored_at = datetime.now().isoformat(timespec="seconds")

    for actual_row in rows:
        player_id = str(actual_row["player_id"])
        player_name = _normalize_player_lookup(str(actual_row["player_name"]))
        game_date = str(actual_row["game_date"])
        opponent_abbr = str(actual_row["opponent_abbr"])
        points = float(actual_row["actual_points"])
        assists = float(actual_row["actual_assists"])
        rebounds = float(actual_row["actual_rebounds"])
        row_matched = False

        for index, tracking_row in enumerate(existing_rows):
            id_match = player_id and tracking_row["player_id"] == player_id
            name_match = player_name and _normalize_player_lookup(tracking_row["player_name"]) == player_name
            if not id_match and not name_match:
                continue
            if game_date and tracking_row["game_date"] != game_date:
                continue
            if opponent_abbr and tracking_row["opponent_abbr"] and tracking_row["opponent_abbr"] != opponent_abbr:
                continue

            actual_result = _market_actual_value(
                tracking_row["market"],
                points=points,
                assists=assists,
                rebounds=rebounds,
            )
            try:
                model_projection = float(tracking_row["model_projection"])
            except ValueError:
                continue

            prediction_error = round(model_projection - actual_result, 1)
            absolute_error = round(abs(prediction_error), 1)
            tolerance = MARKET_TOLERANCE.get(tracking_row["market"], 3.0)
            pick_result, pick_hit = _score_pick_result_for_side(
                edge_value=_parse_optional_float(tracking_row["edge"]),
                line_value=tracking_row["sportsbook_line"],
                actual_result=actual_result,
                pick_side=tracking_row.get("pick_side", ""),
            )

            tracking_row["actual_points"] = f"{points:.1f}"
            tracking_row["actual_assists"] = f"{assists:.1f}"
            tracking_row["actual_rebounds"] = f"{rebounds:.1f}"
            tracking_row["actual_result"] = f"{actual_result:.1f}"
            tracking_row["prediction_error"] = f"{prediction_error:.1f}"
            tracking_row["absolute_error"] = f"{absolute_error:.1f}"
            tracking_row["within_tolerance"] = "1" if absolute_error <= tolerance else "0"
            tracking_row["pick_result"] = pick_result
            tracking_row["pick_hit"] = pick_hit
            tracking_row["scored_at"] = scored_at
            matched_rows += 1
            row_matched = True
            updated_indexes.add(index)

        if row_matched:
            updated_players += 1

    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    with tracking_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRACKING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing_rows)

    return {"matched_rows": matched_rows, "updated_players": updated_players}


def _deduplicate_pick_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """For Underdog rows that have both a More and Less entry for the same
    player/date/market, keep only the one whose pick_side aligns with the
    edge direction. This prevents the hit-rate from being ~50% by construction."""
    seen: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("player_id", ""), row.get("game_date", ""), row.get("market", ""), row.get("sportsbook", ""))
        pick_side = _normalize_pick_side(row.get("pick_side", ""))
        edge = _parse_optional_float(row.get("edge"))
        # Only deduplicate when both More and Less rows exist (Underdog export)
        if pick_side and edge is not None:
            edge_pick = "more" if edge >= 0 else "less"
            if key not in seen:
                seen[key] = row
            else:
                # Keep the row whose pick_side matches the edge direction
                existing_pick = _normalize_pick_side(seen[key].get("pick_side", ""))
                if pick_side == edge_pick and existing_pick != edge_pick:
                    seen[key] = row
        else:
            # No pick_side → use a unique key so it's always included
            unique_key = key + (row.get("created_at", ""), row.get("pick_side", ""))
            seen[unique_key] = row  # type: ignore[assignment]
    return list(seen.values())


def _build_pending_picks() -> list[dict]:
    """Return unscored tracked picks grouped by game_date, most recent first."""
    rows = _ensure_tracking_file_schema()
    pending = [row for row in rows if not row.get("actual_result")]
    by_date: dict[str, list] = {}
    for row in pending:
        game_date = row.get("game_date", "")
        by_date.setdefault(game_date, []).append(row)
    result = []
    for game_date in sorted(by_date.keys(), reverse=True)[:3]:
        date_rows = by_date[game_date]
        players = sorted({r.get("player_name", "") for r in date_rows if r.get("player_name")})
        result.append({
            "game_date": game_date,
            "count": len(date_rows),
            "players": players[:12],
        })
    return result


def _build_player_accuracy() -> list[dict]:
    """Per-player accuracy breakdown from all scored rows."""
    rows = _ensure_tracking_file_schema()
    scored = [row for row in rows if row.get("actual_result")]
    scored = _deduplicate_pick_rows(scored)
    buckets: dict[str, list] = {}
    for row in scored:
        name = row.get("player_name", "Unknown")
        buckets.setdefault(name, []).append(row)
    result = []
    for name, bucket in buckets.items():
        pick_rows = [r for r in bucket if r.get("pick_hit") in {"0", "1"}]
        errors = [float(r["absolute_error"]) for r in bucket if r.get("absolute_error")]
        hits = sum(1 for r in pick_rows if r["pick_hit"] == "1")
        result.append({
            "player_name": name,
            "count": len(bucket),
            "hit_rate": round(hits / len(pick_rows) * 100, 1) if pick_rows else None,
            "avg_error": round(sum(errors) / len(errors), 2) if errors else None,
            "hits": hits,
            "total_picks": len(pick_rows),
        })
    result.sort(key=lambda x: x["count"], reverse=True)
    return result[:30]


def _build_accuracy_summary() -> dict[str, object]:
    rows = _ensure_tracking_file_schema()
    scored_rows = [row for row in rows if row.get("actual_result")]
    scored_rows = _deduplicate_pick_rows(scored_rows)
    if not scored_rows:
        return {
            "scored_rows": 0,
            "player_games": 0,
            "accuracy_rating": "Unscored",
            "overall_mae": None,
            "pick_hit_rate": None,
            "market_summaries": [],
        }

    market_buckets: dict[str, list[dict[str, str]]] = {}
    for row in scored_rows:
        market_buckets.setdefault(row["market"], []).append(row)

    market_summaries = []
    ratio_values: list[float] = []
    all_abs_errors: list[float] = []
    pick_hits = []

    for market, bucket in sorted(market_buckets.items()):
        errors = [float(row["absolute_error"]) for row in bucket if row.get("absolute_error")]
        if not errors:
            continue
        tolerance = MARKET_TOLERANCE.get(market, 3.0)
        within = [row for row in bucket if row.get("within_tolerance") == "1"]
        market_pick_hits = [row for row in bucket if row.get("pick_hit") in {"0", "1"}]
        mae = round(sum(errors) / len(errors), 2)
        within_rate = round((len(within) / len(bucket)) * 100, 1)
        pick_hit_rate = round((sum(1 for row in market_pick_hits if row["pick_hit"] == "1") / len(market_pick_hits)) * 100, 1) if market_pick_hits else None
        market_summaries.append(
            {
                "market": market,
                "count": len(bucket),
                "mae": mae,
                "tolerance": tolerance,
                "within_rate": within_rate,
                "pick_hit_rate": pick_hit_rate,
            }
        )
        ratio_values.append(mae / tolerance)
        all_abs_errors.extend(errors)
        pick_hits.extend(int(row["pick_hit"]) for row in market_pick_hits)

    player_games = {
        (row["player_id"], row["game_date"], row["opponent_abbr"])
        for row in scored_rows
    }
    overall_mae = round(sum(all_abs_errors) / len(all_abs_errors), 2) if all_abs_errors else None
    overall_pick_hit = round((sum(pick_hits) / len(pick_hits)) * 100, 1) if pick_hits else None

    return {
        "scored_rows": len(scored_rows),
        "player_games": len(player_games),
        "accuracy_rating": _accuracy_tier(sum(ratio_values) / len(ratio_values)) if ratio_values else "Unscored",
        "overall_mae": overall_mae,
        "pick_hit_rate": overall_pick_hit,
        "market_summaries": market_summaries,
    }


def _build_prediction_report(
    *,
    game_date: str = "",
    date_from: str = "",
    date_to: str = "",
    market: str = "",
    sportsbook: str = "",
    limit: int = 100,
) -> dict[str, object]:
    rows = _ensure_tracking_file_schema()
    scored_rows = [row for row in rows if row.get("actual_result")]
    if game_date:
        scored_rows = [row for row in scored_rows if row.get("game_date") == game_date]
    if date_from:
        scored_rows = [row for row in scored_rows if str(row.get("game_date", "")) >= date_from]
    if date_to:
        scored_rows = [row for row in scored_rows if str(row.get("game_date", "")) <= date_to]
    if market:
        scored_rows = [row for row in scored_rows if row.get("market") == market]
    if sportsbook:
        scored_rows = [row for row in scored_rows if row.get("sportsbook", "").lower() == sportsbook.lower()]
    scored_rows = _deduplicate_pick_rows(scored_rows)

    scored_rows.sort(
        key=lambda row: (
            str(row.get("game_date", "")),
            str(row.get("created_at", "")),
            str(row.get("player_name", "")),
            str(row.get("market", "")),
        ),
        reverse=True,
    )

    report_rows: list[dict[str, object]] = []
    absolute_errors: list[float] = []
    edges: list[float] = []
    pick_hits: list[int] = []

    for row in scored_rows[: max(limit, 1)]:
        absolute_error = _parse_optional_float(row.get("absolute_error")) or 0.0
        prediction_error = _parse_optional_float(row.get("prediction_error")) or 0.0
        edge_value = _parse_optional_float(row.get("edge"))
        line_value = _parse_optional_float(row.get("sportsbook_line"))
        model_projection = _parse_optional_float(row.get("model_projection")) or 0.0
        actual_result = _parse_optional_float(row.get("actual_result")) or 0.0

        absolute_errors.append(absolute_error)
        if edge_value is not None:
            edges.append(edge_value)
        if row.get("pick_hit") in {"0", "1"}:
            pick_hits.append(int(row["pick_hit"]))

        report_rows.append(
            {
                "game_date": row.get("game_date", ""),
                "player_name": row.get("player_name", ""),
                "opponent_abbr": row.get("opponent_abbr", ""),
                "market": row.get("market", ""),
                "sportsbook": row.get("sportsbook", ""),
                "pick_side": row.get("pick_side", ""),
                "model_projection": round(model_projection, 1),
                "sportsbook_line": line_value,
                "actual_result": round(actual_result, 1),
                "prediction_error": round(prediction_error, 1),
                "absolute_error": round(absolute_error, 1),
                "edge": edge_value,
                "edge_label": row.get("edge_label", ""),
                "pick_result": row.get("pick_result", ""),
                "pick_hit": row.get("pick_hit", ""),
                "within_tolerance": row.get("within_tolerance", ""),
            }
        )

    # P&L — $10 flat entry per pick
    pnl_total = 0.0
    pnl_count = 0
    for row in scored_rows:
        if row.get("pick_hit") not in {"0", "1"}:
            continue
        payout = _parse_optional_float(row.get("payout_multiplier")) or 3.0
        if row["pick_hit"] == "1":
            pnl_total += (payout - 1.0) * 10.0
        else:
            pnl_total -= 10.0
        pnl_count += 1

    # Streak — consecutive hits or misses from most recent row
    streak = 0
    streak_type: str | None = None
    for row in scored_rows:
        if row.get("pick_hit") not in {"0", "1"}:
            continue
        current = "HIT" if row["pick_hit"] == "1" else "MISS"
        if streak_type is None:
            streak_type = current
            streak = 1
        elif current == streak_type:
            streak += 1
        else:
            break

    summary = {
        "total_rows": len(scored_rows),
        "displayed_rows": len(report_rows),
        "avg_abs_error": round(sum(absolute_errors) / len(absolute_errors), 2) if absolute_errors else None,
        "avg_edge": round(sum(abs(edge) for edge in edges) / len(edges), 2) if edges else None,
        "pick_hit_rate": round((sum(pick_hits) / len(pick_hits)) * 100, 1) if pick_hits else None,
        "pnl": round(pnl_total, 2) if pnl_count > 0 else None,
        "pnl_roi": round(pnl_total / (pnl_count * 10.0) * 100, 1) if pnl_count > 0 else None,
        "pnl_count": pnl_count,
        "streak": streak,
        "streak_type": streak_type,
    }
    return {"rows": report_rows, "summary": summary}


def _build_calibration_summary() -> dict[str, dict[str, float | int | str | bool | None]]:
    rows = _ensure_tracking_file_schema()
    scored_rows = [row for row in rows if row.get("actual_result") and row.get("model_projection")]
    calibration = {}
    for market_label, _ in TRACKED_MARKETS:
        bucket = [row for row in scored_rows if row.get("market") == market_label]
        if not bucket:
            calibration[market_label] = {
                "bias": 0.0,
                "sample_count": 0,
                "mae": None,
                "within_rate": None,
                "confidence": "Unscored",
                "applied": False,
            }
            continue

        errors = []
        absolute_errors = []
        within_hits = 0
        for row in bucket:
            try:
                actual_result = float(row["actual_result"])
                model_projection = float(row["model_projection"])
            except ValueError:
                continue
            error = actual_result - model_projection
            errors.append(error)
            absolute_errors.append(abs(error))
            if row.get("within_tolerance") == "1":
                within_hits += 1

        sample_count = len(errors)
        if sample_count == 0:
            calibration[market_label] = {
                "bias": 0.0,
                "sample_count": 0,
                "mae": None,
                "within_rate": None,
                "confidence": "Unscored",
                "applied": False,
            }
            continue

        bias = round(sum(errors) / sample_count, 2)
        mae = round(sum(absolute_errors) / sample_count, 2)
        within_rate = round((within_hits / sample_count) * 100, 1)
        calibration[market_label] = {
            "bias": bias,
            "sample_count": sample_count,
            "mae": mae,
            "within_rate": within_rate,
            "confidence": _confidence_label(within_rate, sample_count),
            "applied": sample_count >= 3,
        }
    return calibration


def _tracked_date_summaries() -> list[dict[str, str | int]]:
    rows = _ensure_tracking_file_schema()
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        game_date = row.get("game_date", "")
        if not game_date:
            continue
        bucket = buckets.setdefault(game_date, {"rows": 0, "scored_rows": 0})
        bucket["rows"] += 1
        if row.get("actual_result"):
            bucket["scored_rows"] += 1
    summaries = [
        {"game_date": game_date, "rows": values["rows"], "scored_rows": values["scored_rows"]}
        for game_date, values in buckets.items()
    ]
    summaries.sort(key=lambda row: str(row["game_date"]), reverse=True)
    return summaries


def _fetch_actual_rows_from_tracking(*, game_date: str) -> tuple[list[dict[str, str | float]], list[str]]:
    tracking_rows = _ensure_tracking_file_schema()
    tracked_dates = _tracked_date_summaries()
    if not game_date:
        if tracked_dates:
            game_date = str(tracked_dates[0]["game_date"])
        else:
            return [], ["Choose a game date to auto-fetch actual results."]

    candidates = {}
    for row in tracking_rows:
        if row.get("game_date") != game_date:
            continue
        key = (row.get("player_id", ""), row.get("player_name", ""), row.get("opponent_abbr", ""))
        candidates[key] = row

    if not candidates:
        available_dates = ", ".join(str(row["game_date"]) for row in tracked_dates[:5])
        message = f"No tracked predictions were found for {game_date}."
        if available_dates:
            message += f" Available tracked dates: {available_dates}."
        return [], [message]

    fetched_rows = []
    errors = []
    for player_id, player_name, opponent_abbr in candidates.keys():
        try:
            actuals = client.get_player_actual_result(
                player_id,
                game_date=game_date,
                opponent_abbr=opponent_abbr or None,
            )
        except Exception as exc:
            print(f"Auto actual fetch error for {player_name or player_id}: {exc}")
            actuals = None
        if actuals is None:
            errors.append(f"Could not auto-fetch actual stats for {player_name or player_id}.")
            continue
        fetched_rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "game_date": game_date,
                "opponent_abbr": opponent_abbr,
                **actuals,
            }
        )
    if not fetched_rows and not errors:
        errors.append("No actual stats were returned for that date.")
    return fetched_rows, errors


def _edge_summary(projection: float, line_value: float | None) -> tuple[float | None, str]:
    if line_value is None:
        return None, "No line"
    edge = round(projection - line_value, 1)
    if edge > 0:
        return edge, f"Over by {edge:.1f}"
    if edge < 0:
        return edge, f"Under by {abs(edge):.1f}"
    return edge, "Right on the number"


def _pick_edge_summary(
    projection: float,
    line_value: float | None,
    *,
    selection_key: str | None = None,
    selection_label: str | None = None,
) -> tuple[float | None, str]:
    if line_value is None:
        return None, "No line"
    display_label = str(selection_label or _pick_side_label(selection_key)).strip() or "Line"
    if selection_key == "less":
        edge = round(line_value - projection, 1)
        if edge > 0:
            return edge, f"{display_label} +{edge:.1f}"
        if edge < 0:
            return edge, f"{display_label} -{abs(edge):.1f}"
        return edge, f"{display_label} 0.0"
    if selection_key == "more":
        edge = round(projection - line_value, 1)
        if edge > 0:
            return edge, f"{display_label} +{edge:.1f}"
        if edge < 0:
            return edge, f"{display_label} -{abs(edge):.1f}"
        return edge, f"{display_label} 0.0"
    return _edge_summary(projection, line_value)


def _edge_class(edge_value: float | None) -> str:
    if edge_value in (None, 0.0):
        return "edge-neutral"
    return "edge-over" if edge_value > 0 else "edge-under"


def _edge_signal(
    market_label: str,
    edge_value: float | None,
    *,
    favor_positive_only: bool = False,
) -> tuple[str, str, str]:
    thresholds = EDGE_SIGNAL_THRESHOLDS.get(market_label, {"lean": 2.0, "best": 3.0})
    lean_threshold = thresholds["lean"]
    best_threshold = thresholds["best"]
    note = f"Lean at {lean_threshold:.1f}+ | Best bet at {best_threshold:.1f}+"
    if edge_value is None:
        return "No line", "signal-pass", note
    if favor_positive_only and edge_value <= 0:
        return "Pass", "signal-pass", note
    absolute_edge = abs(edge_value)
    if absolute_edge >= best_threshold:
        return "Best Bet", "signal-best", note
    if absolute_edge >= lean_threshold:
        return "Lean", "signal-lean", note
    return "Pass", "signal-pass", note


def _build_market_cards(
    prediction: dict[str, float],
    line_inputs: dict[str, float | None],
    calibration_summary: dict[str, dict[str, float | int | str | bool | None]] | None = None,
    confidence_summary: dict[str, dict[str, float | str]] | None = None,
    line_contexts: dict[str, dict[str, float | str | None]] | None = None,
) -> list[dict[str, str | float | None]]:
    cards = []
    for market_label, line_key in TRACKED_MARKETS:
        line_value = line_inputs.get(line_key)
        raw_prediction = prediction[market_label]
        calibration = (calibration_summary or {}).get(market_label, {})
        calibration_bias = float(calibration.get("bias") or 0.0)
        calibration_applied = bool(calibration.get("applied"))
        calibrated_prediction = round(raw_prediction + calibration_bias, 1) if calibration_applied else raw_prediction
        context = (line_contexts or {}).get(line_key, {})
        pick_side = _normalize_pick_side(str(context.get("pick_side") or ""))
        payout_multiplier = _parse_optional_float(str(context.get("payout_multiplier") or "")) if context.get("payout_multiplier") not in (None, "") else None
        edge_value, edge_label = _pick_edge_summary(
            calibrated_prediction,
            line_value,
            selection_key=pick_side or None,
        )
        signal_label, signal_class, signal_note = _edge_signal(
            market_label,
            edge_value,
            favor_positive_only=bool(pick_side),
        )
        market_confidence = (confidence_summary or {}).get(market_label, {})
        cards.append(
            {
                "market": market_label,
                "prediction": calibrated_prediction,
                "raw_prediction": raw_prediction,
                "calibrated_prediction": calibrated_prediction,
                "line": line_value,
                "edge": edge_value,
                "edge_label": edge_label,
                "edge_class": "edge-neutral" if edge_value in (None, 0.0) else ("edge-over" if edge_value > 0 else "edge-under"),
                "signal_label": signal_label,
                "signal_class": signal_class,
                "signal_note": signal_note,
                "pick_side": pick_side,
                "pick_side_label": _pick_side_label(pick_side),
                "payout_multiplier": payout_multiplier,
                "payout_multiplier_label": f"{payout_multiplier:.2f}x" if payout_multiplier is not None else "",
                "calibration_bias": calibration_bias,
                "calibration_applied": calibration_applied,
                "calibration_count": int(calibration.get("sample_count") or 0),
                "calibration_mae": calibration.get("mae"),
                "calibration_confidence": calibration.get("confidence") or "Unscored",
                "raw_edge": _pick_edge_summary(raw_prediction, line_value, selection_key=pick_side or None)[0],
                "model_confidence_score": market_confidence.get("score"),
                "model_confidence_label": market_confidence.get("label") or "Medium",
                "model_error_band": market_confidence.get("error_band"),
            }
        )
    return cards


def _merge_fetched_lines(
    manual_lines: dict[str, float | None],
    fetched_lines: dict[str, float],
) -> dict[str, float | None]:
    merged = dict(manual_lines)
    for key, fetched_value in fetched_lines.items():
        if merged.get(key) is None:
            merged[key] = fetched_value
    return merged


def _empty_line_inputs() -> dict[str, float | None]:
    return {
        "line_points": None,
        "line_assists": None,
        "line_rebounds": None,
        "line_points_rebounds": None,
        "line_points_assists": None,
        "line_assists_rebounds": None,
        "line_points_rebounds_assists": None,
    }


def _build_import_line_contexts(row: dict[str, str | float | None]) -> dict[str, dict[str, float | str | None]]:
    pick_side = _normalize_pick_side(str(row.get("pick_side") or ""))
    payout_multiplier = _parse_optional_float(str(row.get("payout_multiplier") or "")) if row.get("payout_multiplier") not in (None, "") else None
    if not pick_side and payout_multiplier is None:
        return {}

    populated_line_keys = [
        line_key
        for _, line_key in TRACKED_MARKETS
        if row.get(line_key) is not None
    ]
    market_key = str(row.get("market_key") or "").strip()
    if market_key:
        target_keys = [market_key]
    elif len(populated_line_keys) == 1:
        target_keys = populated_line_keys
    else:
        return {}

    return {
        target_key: {
            "pick_side": pick_side,
            "payout_multiplier": payout_multiplier,
        }
        for target_key in target_keys
    }


def _fetch_slate_lines(
    *,
    player_name: str,
    opponent_abbr: str,
    game_date: str,
    line_sources: list[tuple[str, object]],
) -> tuple[dict[str, float | None], str]:
    merged_lines = _empty_line_inputs()
    matched_sources: list[str] = []

    for source_name, source_client in line_sources:
        try:
            fetched_lines = source_client.fetch_player_lines(
                player_name=player_name,
                opponent_abbr=opponent_abbr,
                game_date=game_date or None,
            )
        except Exception as exc:
            print(f"{source_name} slate line lookup error for {player_name}: {exc}")
            fetched_lines = {}
        if fetched_lines:
            merged_lines = _merge_fetched_lines(merged_lines, fetched_lines)
            matched_sources.append(source_name)

    return merged_lines, ", ".join(matched_sources)


def _extract_slate_line_inputs(player_id: int, values) -> dict[str, float | None]:
    return {
        "line_points": _parse_optional_float(values.get(f"{player_id}_line_points")),
        "line_assists": _parse_optional_float(values.get(f"{player_id}_line_assists")),
        "line_rebounds": _parse_optional_float(values.get(f"{player_id}_line_rebounds")),
        "line_points_rebounds": _parse_optional_float(values.get(f"{player_id}_line_points_rebounds")),
        "line_points_assists": _parse_optional_float(values.get(f"{player_id}_line_points_assists")),
        "line_assists_rebounds": _parse_optional_float(values.get(f"{player_id}_line_assists_rebounds")),
        "line_points_rebounds_assists": _parse_optional_float(values.get(f"{player_id}_line_points_rebounds_assists")),
    }


def _board_context(
    *,
    title: str,
    source_name: str,
    message: str,
    rows: list[dict[str, str | float | None]],
    min_edge: float,
    total_lines: int,
    total_players: int,
    displayed_players: int,
    matched_players: int,
    unmatched_players: int,
    page: int,
    total_pages: int,
    market_filters: list[str] | None = None,
    selected_market: str = "all",
    parlay_combos: dict | None = None,
    line_movers: list | None = None,
    latest_snapshot_at: str | None = None,
    book_discrepancies: list | None = None,
):
    page_window_start = max(1, page - 2)
    page_window_end = min(total_pages, page + 2)
    return render_template(
        'prizepicks_board.html',
        board_title=title,
        board_source=source_name,
        line_label=f"{source_name} Line",
        edge_label="Pick Edge" if source_name in {"ParlayPlay", "Underdog"} else "Edge",
        show_pick_columns=source_name in {"ParlayPlay", "Underdog"} or any(
            row.get("selection_label") or row.get("payout_multiplier_label")
            for row in rows
        ),
        board_rows=rows,
        filters={"min_edge": min_edge, "market": selected_market},
        board_summary={
            "total_lines": total_lines,
            "total_players": total_players,
            "displayed_players": displayed_players,
            "matched_players": matched_players,
            "displayed_rows": len(rows),
            "unmatched_players": unmatched_players,
            "message": message,
        },
        pagination={
            "page": page,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "prev_page": page - 1,
            "next_page": page + 1,
            "page_numbers": list(range(page_window_start, page_window_end + 1)),
            "players_per_page": BOARD_PLAYERS_PER_PAGE,
        },
        market_filters=market_filters or [],
        selected_market=selected_market,
        parlay_combos=parlay_combos or {},
        line_movers=line_movers or [],
        latest_snapshot_at=latest_snapshot_at,
        book_discrepancies=book_discrepancies or [],
    )


def _build_parlay_combos(
    board_rows: list[dict],
    sizes: tuple[int, ...] = (2, 3, 4, 5),
    candidate_pool: int = 25,
) -> dict[int, list[dict]]:
    """Best N-pick combos by max total edge, no duplicate players per combo."""
    from itertools import combinations

    def _confidence_score(r: dict) -> float:
        hit_rate = MARKET_HIT_RATE.get(r.get("market", ""), 0.5)
        return float(r.get("absolute_edge", 0.0)) * hit_rate

    eligible = sorted(
        [r for r in board_rows if float(r.get("edge") or 0.0) > 0],
        key=_confidence_score,
        reverse=True,
    )[:candidate_pool]

    result: dict[int, list[dict]] = {}
    for size in sizes:
        if len(eligible) < size:
            result[size] = []
            continue
        best_score = -1.0
        best_combo: list[dict] = []
        for combo in combinations(eligible, size):
            if len({str(r["player_id"]) for r in combo}) < size:
                continue
            score = sum(_confidence_score(r) for r in combo)
            if score > best_score:
                best_score = score
                best_combo = list(combo)
        result[size] = best_combo
    return result


def _paginate_board_rows(
    rows: list[dict[str, str | float | None]],
    *,
    page: int,
    players_per_page: int = BOARD_PLAYERS_PER_PAGE,
) -> dict[str, int | list[dict[str, str | float | None]]]:
    grouped_rows: dict[tuple[str, str, str, str], list[dict[str, str | float | None]]] = {}
    ordered_player_keys: list[tuple[str, str, str, str]] = []

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        player_key = (
            str(row.get("player_id") or ""),
            str(row.get("player_name") or ""),
            str(row.get("opponent_abbr") or ""),
            str(row.get("game_date") or ""),
        )
        if player_key not in grouped_rows:
            grouped_rows[player_key] = []
            ordered_player_keys.append(player_key)
        grouped_rows[player_key].append(row)

    total_players = len(ordered_player_keys)
    total_pages = max(1, (total_players + players_per_page - 1) // players_per_page) if total_players else 1
    current_page = min(max(page, 1), total_pages)
    start_index = (current_page - 1) * players_per_page
    end_index = start_index + players_per_page
    page_player_keys = ordered_player_keys[start_index:end_index]

    paged_rows: list[dict[str, str | float | None]] = []
    for player_key in page_player_keys:
        paged_rows.extend(grouped_rows[player_key])

    return {
        "rows": paged_rows,
        "page": current_page,
        "total_pages": total_pages,
        "total_players": total_players,
        "displayed_players": len(page_player_keys),
    }


@lru_cache(maxsize=512)
def _resolve_player_id_cached(player_name: str) -> int | None:
    return client.resolve_player_id_by_name(player_name)


@lru_cache(maxsize=2048)
def _cached_prediction_triplet(
    player_id: str,
    opponent_abbr: str,
    game_date: str,
) -> tuple[float, float, float, object]:
    result = predict_player_statline(
        player_id,
        opponent_abbr=opponent_abbr or None,
        game_date=game_date or None,
    )
    return (result["points"], result["assists"], result["rebounds"], result.get("confirmed_starter"), result.get("expected_minutes", 0.0))


def _get_prediction_summary_cached(
    *,
    player_id: str,
    opponent_abbr: str,
    game_date: str,
) -> dict[str, float]:
    points, assists, rebounds, confirmed_starter, expected_minutes = _cached_prediction_triplet(
        player_id,
        opponent_abbr,
        game_date,
    )
    summary = _build_prediction_summary(points, assists, rebounds)
    summary["confirmed_starter"] = confirmed_starter
    summary["expected_minutes"] = expected_minutes
    return summary


@lru_cache(maxsize=1)
def _cached_underdog_board_snapshot() -> dict[str, object]:
    board_entries = underdog_client.fetch_board_entries()
    prediction_cache: dict[tuple[str, str, str], dict[str, float]] = {}
    unmatched_players: set[str] = set()
    board_rows: list[dict[str, str | float | None]] = []
    resolved_entries: list[tuple[object, int, str, str, tuple[str, str, str]]] = []
    unique_prediction_keys: dict[tuple[str, str, str], str] = {}

    for entry in board_entries:
        player_id = _resolve_player_id_cached(entry.player_name)
        if player_id is None:
            unmatched_players.add(entry.player_name)
            continue

        game_date = _parse_start_date(entry.start_time)
        opponent_abbr = entry.opponent_abbr if entry.opponent_abbr in NBA_TEAM_OPTIONS else ""
        cache_key = (str(player_id), opponent_abbr, game_date)
        resolved_entries.append((entry, player_id, opponent_abbr, game_date, cache_key))
        unique_prediction_keys.setdefault(cache_key, entry.player_name)

    if unique_prediction_keys:
        max_workers = min(UNDERDOG_BOARD_PREDICTION_WORKERS, len(unique_prediction_keys))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _get_prediction_summary_cached,
                    player_id=cache_key[0],
                    opponent_abbr=cache_key[1],
                    game_date=cache_key[2],
                ): cache_key
                for cache_key in unique_prediction_keys
            }
            for future in as_completed(future_map):
                cache_key = future_map[future]
                player_name = unique_prediction_keys[cache_key]
                try:
                    prediction_cache[cache_key] = future.result()
                except Exception as exc:
                    print(f"Underdog board prediction error for {player_name}: {exc}")

    for entry, player_id, opponent_abbr, game_date, cache_key in resolved_entries:
        prediction = prediction_cache.get(cache_key)
        if prediction is None:
            continue

        prediction = prediction_cache[cache_key]
        if float(prediction.get("expected_minutes") or 0.0) < 15.0:
            continue
        model_projection = prediction[entry.market_label]
        edge_value, edge_label = _pick_edge_summary(
            model_projection,
            entry.line_score,
            selection_key=entry.selection_key,
            selection_label=entry.selection_label,
        )
        signal_label, signal_class, signal_note = _edge_signal(
            entry.market_label,
            edge_value,
            favor_positive_only=bool(entry.selection_key),
        )
        confirmed_starter = prediction.get("confirmed_starter")
        high_confidence_assists = bool(
            entry.market_label == "Assists"
            and edge_value is not None and edge_value >= 1.0
            and (entry.selection_label or "").lower() in ("higher", "over", "line")
            and entry.line_score < 5.0
            and confirmed_starter is True
        )
        board_rows.append(
            {
                "player_name": entry.player_name,
                "player_id": player_id,
                "market": entry.market_label,
                "selection_label": entry.selection_label or "Line",
                "payout_multiplier": entry.payout_multiplier,
                "payout_multiplier_label": f"{entry.payout_multiplier:.2f}x" if entry.payout_multiplier is not None else "N/A",
                "opponent_abbr": opponent_abbr or "N/A",
                "game_date": game_date or "N/A",
                "sportsbook_line": round(entry.line_score, 1),
                "model_projection": model_projection,
                "edge": edge_value,
                "edge_label": edge_label,
                "edge_class": _edge_class(edge_value),
                "signal_label": signal_label,
                "signal_class": signal_class,
                "signal_note": signal_note,
                "absolute_edge": abs(edge_value) if edge_value is not None else 0.0,
                "confirmed_starter": confirmed_starter,
                "high_confidence_assists": high_confidence_assists,
            }
        )

    # For each player/market/date group keep only the edge-aligned pick (positive edge).
    # Underdog sends both Higher and Lower for every line; only one direction is ever
    # useful and showing both confuses the board.
    deduped: dict[tuple, dict] = {}
    for row in board_rows:
        key = (str(row.get("player_id", "")), str(row.get("market", "")), str(row.get("game_date", "")))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
        else:
            # Prefer the row whose edge is positive (pick agrees with model)
            row_edge = float(row.get("edge") or 0.0)
            existing_edge = float(existing.get("edge") or 0.0)
            if row_edge > existing_edge:
                deduped[key] = row
    board_rows = list(deduped.values())

    board_rows.sort(key=lambda row: float(row["absolute_edge"]), reverse=True)
    return {
        "board_rows": tuple(board_rows),
        "total_lines": len(board_entries),
        "matched_players": len(prediction_cache),
        "unmatched_players": len(unmatched_players),
    }


@lru_cache(maxsize=1)
def _cached_ncaab_board_snapshot() -> dict[str, object]:
    from ncaa_prediction import predict_player_statline as _ncaa_predict_statline
    board_entries = underdog_client.fetch_board_entries(sport="ncaab")
    prediction_cache: dict[tuple[str, str], dict[str, float]] = {}
    unmatched_players: set[str] = set()
    board_rows: list[dict] = []
    unique_keys: dict[tuple[str, str], str] = {}

    for entry in board_entries:
        game_date = _parse_start_date(entry.start_time)
        opponent = entry.opponent_abbr or ""
        cache_key = (entry.player_name.lower(), opponent.upper())
        unique_keys.setdefault(cache_key, entry.player_name)

    max_workers = min(UNDERDOG_BOARD_PREDICTION_WORKERS, max(len(unique_keys), 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        def _fetch_ncaa(key_name_pair):
            key, player_name = key_name_pair
            _, opponent = key
            try:
                result = _ncaa_predict_statline(player_name, opponent=opponent or None)
                summary = _build_prediction_summary(
                    float(result["points"]),
                    float(result["assists"]),
                    float(result["rebounds"]),
                )
                return key, summary
            except Exception:
                return key, None

        for key, summary in executor.map(_fetch_ncaa, unique_keys.items()):
            if summary is not None:
                prediction_cache[key] = summary
            else:
                unmatched_players.add(unique_keys[key])

    for entry in board_entries:
        opponent = entry.opponent_abbr or ""
        cache_key = (entry.player_name.lower(), opponent.upper())
        prediction = prediction_cache.get(cache_key)

        model_projection = prediction.get(entry.market_label) if prediction else None

        game_date = _parse_start_date(entry.start_time)
        edge_value, edge_label = _pick_edge_summary(
            model_projection,
            entry.line_score,
            selection_key=entry.selection_key,
            selection_label=entry.selection_label,
        )
        signal_label, signal_class, signal_note = _edge_signal(
            entry.market_label,
            edge_value,
            favor_positive_only=bool(entry.selection_key),
        )
        board_rows.append({
            "player_name": entry.player_name,
            "player_id": entry.player_name.lower(),
            "market": entry.market_label,
            "selection_label": entry.selection_label or "Line",
            "payout_multiplier": entry.payout_multiplier,
            "payout_multiplier_label": f"{entry.payout_multiplier:.2f}x" if entry.payout_multiplier is not None else "N/A",
            "opponent_abbr": opponent or "N/A",
            "game_date": game_date or "N/A",
            "sportsbook_line": round(entry.line_score, 1),
            "model_projection": model_projection,
            "edge": edge_value,
            "edge_label": edge_label,
            "edge_class": _edge_class(edge_value),
            "signal_label": signal_label,
            "signal_class": signal_class,
            "signal_note": signal_note,
            "absolute_edge": abs(edge_value) if edge_value is not None else 0.0,
            "confirmed_starter": None,
            "high_confidence_assists": False,
        })

    deduped: dict[tuple, dict] = {}
    for row in board_rows:
        key = (str(row["player_id"]), str(row["market"]), str(row["game_date"]))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
        else:
            if float(row.get("edge") or 0.0) > float(existing.get("edge") or 0.0):
                deduped[key] = row
    board_rows = sorted(deduped.values(), key=lambda r: float(r["absolute_edge"]), reverse=True)

    return {
        "board_rows": tuple(board_rows),
        "total_lines": len(board_entries),
        "matched_players": len(prediction_cache),
        "unmatched_players": len(unmatched_players),
    }


def _prewarm_underdog_board_cache() -> None:
    if not underdog_client.is_configured():
        return
    try:
        _cached_underdog_board_snapshot()
    except Exception as exc:
        print(f"Underdog board prewarm error: {exc}")


def _start_underdog_board_prewarm() -> None:
    global _underdog_prewarm_started

    with _underdog_prewarm_lock:
        if _underdog_prewarm_started:
            return
        _underdog_prewarm_started = True

    Thread(target=_prewarm_underdog_board_cache, daemon=True).start()


def _append_tracking_rows(
    *,
    player_id: str,
    player_name: str,
    opponent_abbr: str,
    game_date: str,
    sportsbook: str,
    market_cards: list[dict[str, str | float | None]],
) -> int:
    tracking_rows = []
    created_at = datetime.now().isoformat(timespec="seconds")
    for card in market_cards:
        if card["line"] is None:
            continue
        tracking_rows.append(
            {
                "created_at": created_at,
                "sportsbook": sportsbook,
                "player_id": str(player_id),
                "player_name": player_name,
                "opponent_abbr": opponent_abbr,
                "game_date": game_date,
                "market": str(card["market"]),
                "model_projection": f"{float(card.get('raw_prediction', card['prediction'])):.1f}",
                "sportsbook_line": f"{float(card['line']):.1f}",
                "pick_side": str(card.get("pick_side_label") or ""),
                "payout_multiplier": (
                    f"{float(card['payout_multiplier']):.2f}"
                    if card.get("payout_multiplier") is not None
                    else ""
                ),
                "edge": f"{float(card['edge']):.1f}",
                "edge_label": str(card["edge_label"]),
                "data_mode": current_data_mode()["label"],
                "actual_points": "",
                "actual_assists": "",
                "actual_rebounds": "",
                "actual_result": "",
                "prediction_error": "",
                "absolute_error": "",
                "within_tolerance": "",
                "pick_result": "",
                "pick_hit": "",
                "scored_at": "",
            }
        )

    return _write_tracking_rows(tracking_rows)


def _write_tracking_rows(tracking_rows: list[dict[str, str]]) -> int:
    if not tracking_rows:
        return 0

    tracking_path = settings.tracking_file
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = TRACKING_FIELDNAMES

    existing_keys: set[tuple[str, ...]] = set()
    for row in _ensure_tracking_file_schema():
        existing_keys.add(
            (
                row.get("sportsbook", ""),
                row.get("player_id", ""),
                row.get("opponent_abbr", ""),
                row.get("game_date", ""),
                row.get("market", ""),
                row.get("model_projection", ""),
                row.get("sportsbook_line", ""),
                row.get("pick_side", ""),
                row.get("payout_multiplier", ""),
            )
        )

    new_rows = [
        row
        for row in tracking_rows
        if (
            row["sportsbook"],
            row["player_id"],
            row["opponent_abbr"],
            row["game_date"],
            row["market"],
            row["model_projection"],
            row["sportsbook_line"],
            row["pick_side"],
            row["payout_multiplier"],
        )
        not in existing_keys
    ]
    if not new_rows:
        return 0

    file_exists = tracking_path.exists()
    with tracking_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows)


def _build_tracking_rows_from_board_rows(
    rows: list[dict[str, str | float | None]],
    *,
    sportsbook: str,
) -> list[dict[str, str]]:
    created_at = datetime.now().isoformat(timespec="seconds")
    tracking_rows: list[dict[str, str]] = []

    for row in rows:
        line_value = _parse_optional_float(row.get("sportsbook_line"))
        model_projection = _parse_optional_float(row.get("model_projection"))
        edge_value = _parse_optional_float(row.get("edge"))
        payout_multiplier = _parse_optional_float(row.get("payout_multiplier"))
        if line_value is None or model_projection is None or edge_value is None:
            continue

        tracking_rows.append(
            {
                "created_at": created_at,
                "sportsbook": sportsbook,
                "player_id": str(row.get("player_id") or ""),
                "player_name": str(row.get("player_name") or ""),
                "opponent_abbr": str(row.get("opponent_abbr") or ""),
                "game_date": str(row.get("game_date") or ""),
                "market": str(row.get("market") or ""),
                "model_projection": f"{model_projection:.1f}",
                "sportsbook_line": f"{line_value:.1f}",
                "pick_side": _pick_side_label(str(row.get("selection_label") or "")),
                "payout_multiplier": f"{payout_multiplier:.2f}" if payout_multiplier is not None else "",
                "edge": f"{edge_value:.1f}",
                "edge_label": str(row.get("edge_label") or ""),
                "data_mode": current_data_mode()["label"],
                "actual_points": "",
                "actual_assists": "",
                "actual_rebounds": "",
                "actual_result": "",
                "prediction_error": "",
                "absolute_error": "",
                "within_tolerance": "",
                "pick_result": "",
                "pick_hit": "",
                "scored_at": "",
            }
        )

    return tracking_rows


@app.context_processor
def inject_ui_context():
    return {
        "data_mode": current_data_mode(),
        "team_options": NBA_TEAM_OPTIONS,
        "parlayplay_status": {
            "configured": parlayplay_client.is_configured(),
            "message": parlayplay_client.configuration_message(),
            "hint": parlayplay_client.setup_hint(),
        },
        "underdog_status": {
            "configured": underdog_client.is_configured(),
            "message": underdog_client.configuration_message(),
            "hint": underdog_client.setup_hint(),
        },
    }


@app.route('/')
def home():
    _start_underdog_board_prewarm()
    return render_template('index.html')


@app.route('/import-lines', methods=['GET', 'POST'])
def import_lines():
    sample_csv = (
        "player_name,opponent_abbr,game_date,sportsbook,market,line,pick_side,multiplier\n"
        "Christian Braun,LAL,2026-03-15,ParlayPlay,Points,11.5,Less,1.83\n"
        "Christian Braun,LAL,2026-03-15,ParlayPlay,Points,11.5,More,1.80\n"
        "Jalen Brunson,BOS,2026-03-15,PrizePicks,Points,27.5,,\n"
    )
    context = {
        "sample_csv": sample_csv,
        "default_values": {
            "sportsbook": "PrizePicks",
            "game_date": "",
            "opponent_abbr": "",
            "preview_limit": "10",
            "csv_text": sample_csv,
        },
        "import_summary": None,
        "imported_rows": [],
        "import_errors": [],
        "import_highlights": [],
    }
    if request.method == "GET":
        return render_template('import_lines.html', **context)

    default_sportsbook = request.form.get("default_sportsbook", "").strip() or "Manual Import"
    default_game_date = request.form.get("default_game_date", "").strip()
    default_opponent_abbr = request.form.get("default_opponent_abbr", "").strip().upper()
    preview_limit_raw = request.form.get("preview_limit", "").strip()
    preview_limit = _parse_optional_int(preview_limit_raw)
    csv_text = _read_import_text()
    parsed_rows, import_errors = _parse_import_rows(
        csv_text,
        default_sportsbook=default_sportsbook,
        default_game_date=default_game_date,
        default_opponent_abbr=default_opponent_abbr,
    )
    # When the same player/market/date appears twice (Higher + Lower from Underdog),
    # drop the pick_side so the model determines the right direction rather than
    # logging both rows and cluttering tracking with both directions.
    _dedup_seen: dict[tuple, int] = {}
    for i, row in enumerate(parsed_rows):
        key = (
            str(row.get("player_name") or row.get("player_id") or "").lower().strip(),
            str(row.get("game_date") or ""),
            str(row.get("sportsbook") or ""),
            # use the first non-None line field as market fingerprint
            next(
                (str(row.get(k)) for k in ("line_points", "line_assists", "line_rebounds",
                 "line_points_rebounds", "line_points_assists", "line_assists_rebounds",
                 "line_points_rebounds_assists") if row.get(k) is not None),
                "",
            ),
        )
        if key in _dedup_seen:
            parsed_rows[_dedup_seen[key]]["pick_side"] = None  # clear so model picks direction
            parsed_rows[i] = None  # mark for removal
        else:
            _dedup_seen[key] = i
    parsed_rows = [r for r in parsed_rows if r is not None]
    total_candidate_rows = len(parsed_rows)
    preview_applied = False
    if preview_limit is not None:
        if preview_limit <= 0:
            import_errors.append("Preview limit must be greater than 0.")
        elif total_candidate_rows > preview_limit:
            parsed_rows = parsed_rows[:preview_limit]
            preview_applied = True

    imported_rows: list[dict[str, object]] = []
    total_logged_rows = 0
    processed_count = 0
    matched_players = 0
    import_highlights: list[dict[str, object]] = []

    for row in parsed_rows:
        player_id = str(row["player_id"] or "").strip()
        player_name = str(row["player_name"] or "").strip()
        if not player_id:
            resolved_player_id = client.resolve_player_id_by_name(player_name)
            if resolved_player_id is None:
                import_errors.append(f"Could not match player '{player_name}'.")
                continue
            player_id = str(resolved_player_id)
        if not player_name:
            player = client.get_player_details(player_id)
            player_name = (
                f"{player.get('firstname', '')} {player.get('lastname', '')}".strip()
                if player else f"Player {player_id}"
            )

        if not row["opponent_abbr"] and row.get("team_abbr") and row.get("game_date"):
            inferred_opponent = _infer_import_opponent_abbr(
                player_id=player_id,
                player_name=player_name,
                team_abbr=str(row.get("team_abbr") or ""),
                game_date=str(row.get("game_date") or ""),
            )
            if inferred_opponent:
                row["opponent_abbr"] = inferred_opponent

        line_inputs = {
            "line_points": row["line_points"],
            "line_assists": row["line_assists"],
            "line_rebounds": row["line_rebounds"],
            "line_points_rebounds": row["line_points_rebounds"],
            "line_points_assists": row["line_points_assists"],
            "line_assists_rebounds": row["line_assists_rebounds"],
            "line_points_rebounds_assists": row["line_points_rebounds_assists"],
        }
        line_contexts = _build_import_line_contexts(row)
        try:
            prediction_payload = predict_player_statline(
                player_id,
                opponent_abbr=str(row["opponent_abbr"] or "") or None,
                game_date=str(row["game_date"] or "") or None,
            )
        except Exception as exc:
            print(f"Bulk import prediction error for {player_name}: {exc}")
            import_errors.append(f"Prediction failed for '{player_name}'.")
            continue

        prediction = _build_prediction_summary(
            prediction_payload["points"],
            prediction_payload["assists"],
            prediction_payload["rebounds"],
        )
        confidence_summary = _build_market_confidence_summary(prediction_payload)
        market_cards = [
            card
            for card in _build_market_cards(
                prediction,
                line_inputs,
                confidence_summary=confidence_summary,
                line_contexts=line_contexts,
            )
            if card["line"] is not None
        ]
        logged_rows = _append_tracking_rows(
            player_id=player_id,
            player_name=player_name,
            opponent_abbr=str(row["opponent_abbr"] or ""),
            game_date=str(row["game_date"] or ""),
            sportsbook=str(row["sportsbook"] or "Manual Import"),
            market_cards=market_cards,
        )
        total_logged_rows += logged_rows
        processed_count += 1
        matched_players += 1
        imported_rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "sportsbook": row["sportsbook"],
                "opponent_abbr": row["opponent_abbr"] or "N/A",
                "game_date": row["game_date"] or "N/A",
                "market_cards": market_cards,
                "best_card": market_cards[0] if market_cards else None,
            }
        )
        for card in market_cards:
            import_highlights.append(
                {
                    "player_name": player_name,
                    "sportsbook": row["sportsbook"],
                    "market": card["market"],
                    "pick_side_label": card.get("pick_side_label", ""),
                    "payout_multiplier_label": card.get("payout_multiplier_label", ""),
                    "line": card["line"],
                    "prediction": card["prediction"],
                    "edge_label": card["edge_label"],
                    "edge_class": card["edge_class"],
                    "signal_label": card["signal_label"],
                    "signal_class": card["signal_class"],
                    "absolute_edge": abs(float(card["edge"] or 0.0)),
                }
            )

    context["default_values"] = {
        "sportsbook": default_sportsbook,
        "game_date": default_game_date,
        "opponent_abbr": default_opponent_abbr,
        "preview_limit": preview_limit_raw,
        "csv_text": csv_text or sample_csv,
    }
    context["imported_rows"] = imported_rows
    context["import_errors"] = import_errors
    import_highlights.sort(
        key=lambda row: (
            0 if row["signal_label"] == "Best Bet" else (1 if row["signal_label"] == "Lean" else 2),
            -float(row["absolute_edge"]),
        )
    )
    context["import_highlights"] = import_highlights[:12]
    context["import_summary"] = {
        "processed_count": processed_count,
        "matched_players": matched_players,
        "logged_rows": total_logged_rows,
        "total_candidate_rows": total_candidate_rows,
        "preview_limit": preview_limit,
        "preview_applied": preview_applied,
        "tracking_file": str(settings.tracking_file),
    }
    return render_template('import_lines.html', **context)


@app.route('/accuracy-review', methods=['GET', 'POST'])
def accuracy_review():
    sample_csv = "player_name,opponent_abbr,game_date,actual_points,actual_assists,actual_rebounds\n"
    report_game_date = request.values.get("report_game_date", "").strip()
    report_date_from = request.values.get("report_date_from", "").strip()
    report_date_to = request.values.get("report_date_to", "").strip()
    report_market = request.values.get("report_market", "").strip()
    report_sportsbook = request.values.get("report_sportsbook", "").strip()
    report_limit = _parse_optional_int(request.values.get("report_limit", "").strip()) or 100
    report_limit = max(10, min(report_limit, 500))
    report_payload = _build_prediction_report(
        game_date=report_game_date,
        date_from=report_date_from,
        date_to=report_date_to,
        market=report_market,
        sportsbook=report_sportsbook,
        limit=report_limit,
    )
    context = {
        "default_values": {
            "game_date": "",
            "opponent_abbr": "",
            "csv_text": "",
        },
        "sample_csv": sample_csv,
        "review_errors": [],
        "review_summary": None,
        "accuracy_summary": _build_accuracy_summary(),
        "result_source": "auto",
        "tracked_dates": _tracked_date_summaries(),
        "pending_picks": _build_pending_picks(),
        "player_accuracy": _build_player_accuracy(),
        "report_rows": report_payload["rows"],
        "report_summary": report_payload["summary"],
        "report_filters": {
            "game_date": report_game_date,
            "date_from": report_date_from,
            "date_to": report_date_to,
            "market": report_market,
            "sportsbook": report_sportsbook,
            "limit": report_limit,
        },
        "report_market_options": [market for market, _ in TRACKED_MARKETS],
    }

    if request.method == "GET":
        return render_template('accuracy_review.html', **context)

    default_game_date = request.form.get("default_game_date", "").strip()
    default_opponent_abbr = request.form.get("default_opponent_abbr", "").strip().upper()
    fetch_mode = request.form.get("result_source", "upload")
    quick_action = request.form.get("quick_action", "").strip()
    if quick_action == "yesterday":
        default_game_date = (datetime.now().date() - timedelta(days=1)).isoformat()
        fetch_mode = "auto"
    elif quick_action == "today":
        default_game_date = datetime.now().date().isoformat()
        fetch_mode = "auto"
    csv_text = _read_import_text()
    if fetch_mode == "auto":
        parsed_rows, review_errors = _fetch_actual_rows_from_tracking(game_date=default_game_date)
    else:
        parsed_rows, review_errors = _parse_actual_rows(
            csv_text,
            default_game_date=default_game_date,
            default_opponent_abbr=default_opponent_abbr,
        )
    apply_summary = {"matched_rows": 0, "updated_players": 0}
    if parsed_rows:
        apply_summary = _apply_actual_results(parsed_rows)
        if apply_summary["matched_rows"] == 0:
            review_errors.append("No tracked prediction rows matched those post-game results.")

    context["default_values"] = {
        "game_date": default_game_date,
        "opponent_abbr": default_opponent_abbr,
        "csv_text": csv_text,
    }
    context["review_errors"] = review_errors
    context["review_summary"] = apply_summary
    context["accuracy_summary"] = _build_accuracy_summary()
    context["result_source"] = fetch_mode
    context["tracked_dates"] = _tracked_date_summaries()
    context["pending_picks"] = _build_pending_picks()
    context["player_accuracy"] = _build_player_accuracy()
    report_game_date = report_game_date or default_game_date
    report_payload = _build_prediction_report(
        game_date=report_game_date,
        date_from=report_date_from,
        date_to=report_date_to,
        market=report_market,
        sportsbook=report_sportsbook,
        limit=report_limit,
    )
    context["report_rows"] = report_payload["rows"]
    context["report_summary"] = report_payload["summary"]
    context["report_filters"] = {
        "game_date": report_game_date,
        "date_from": report_date_from,
        "date_to": report_date_to,
        "market": report_market,
        "sportsbook": report_sportsbook,
        "limit": report_limit,
    }
    return render_template('accuracy_review.html', **context)


@app.route('/historical-backtest', methods=['GET', 'POST'])
def historical_backtest():
    overview = get_historical_backtest_overview()
    target_date = request.form.get("target_date", "").strip() if request.method == "POST" else (overview.max_date or "")
    backtest_result = None
    backtest_error = ""
    try:
        if request.method == "POST" and target_date:
            backtest_result = run_historical_backtest(target_date)
    except FileNotFoundError as exc:
        backtest_error = str(exc)
    except Exception as exc:
        print(f"Historical backtest error: {exc}")
        backtest_error = "Unable to run the historical backtest right now."

    return render_template(
        'historical_backtest.html',
        target_date=target_date,
        backtest_result=backtest_result,
        backtest_error=backtest_error,
        backtest_overview=overview,
    )


@app.route('/historical-backtest/batch', methods=['GET', 'POST'])
def batch_backtest():
    overview = get_historical_backtest_overview()
    start_date = request.form.get('start_date', '').strip()
    end_date = request.form.get('end_date', '').strip()
    batch_result = None
    batch_error = ''
    if request.method == 'POST' and start_date and end_date:
        try:
            batch_result = run_batch_backtest(start_date, end_date)
        except Exception as exc:
            print(f'Batch backtest error: {exc}')
            batch_error = 'Unable to run batch backtest right now.'
    return render_template(
        'batch_backtest.html',
        start_date=start_date,
        end_date=end_date,
        batch_result=batch_result,
        batch_error=batch_error,
        overview=overview,
    )


@app.route('/model-insights')
def model_insights():
    insights_error = ""
    insights = None
    try:
        insights = get_model_insights()
    except Exception as exc:
        print(f"Model insights error: {exc}")
        insights_error = "Unable to load model insights right now."

    return render_template(
        'model_insights.html',
        insights=insights,
        insights_error=insights_error,
    )


@app.route('/prizepicks-board')
def prizepicks_board():
    page = max(_parse_optional_int(request.args.get("page")) or 1, 1)
    min_edge = max(_parse_optional_float(request.args.get("min_edge")) or 0.0, 0.0)

    try:
        board_entries = prizepicks_client.fetch_board_entries()
    except PrizePicksProviderError as exc:
        return _board_context(
            title="PrizePicks Edge Board",
            source_name="PrizePicks",
            message=str(exc),
            rows=[],
            min_edge=min_edge,
            total_lines=0,
            total_players=0,
            displayed_players=0,
            matched_players=0,
            unmatched_players=0,
            page=1,
            total_pages=1,
        )
    except Exception as exc:
        print(f"PrizePicks board error: {exc}")
        return _board_context(
            title="PrizePicks Edge Board",
            source_name="PrizePicks",
            message="PrizePicks board is unavailable right now.",
            rows=[],
            min_edge=min_edge,
            total_lines=0,
            total_players=0,
            displayed_players=0,
            matched_players=0,
            unmatched_players=0,
            page=1,
            total_pages=1,
        )

    prediction_cache: dict[tuple[str, str, str], dict[str, float]] = {}
    unmatched_players: set[str] = set()
    board_rows: list[dict[str, str | float | None]] = []

    for entry in board_entries:
        player_id = client.resolve_player_id_by_name(entry.player_name)
        if player_id is None:
            unmatched_players.add(entry.player_name)
            continue

        game_date = _parse_start_date(entry.start_time)
        opponent_abbr = entry.opponent_abbr if entry.opponent_abbr in NBA_TEAM_OPTIONS else ""
        cache_key = (str(player_id), opponent_abbr, game_date)
        if cache_key not in prediction_cache:
            try:
                points, assists, rebounds = predict_player_stats(
                    str(player_id),
                    opponent_abbr=opponent_abbr or None,
                    game_date=game_date or None,
                )
                prediction_cache[cache_key] = _build_prediction_summary(points, assists, rebounds)
            except Exception as exc:
                print(f"PrizePicks board prediction error for {entry.player_name}: {exc}")
                continue

        prediction = prediction_cache[cache_key]
        model_projection = prediction[entry.market_label]
        edge_value, edge_label = _edge_summary(model_projection, entry.line_score)
        signal_label, signal_class, signal_note = _edge_signal(entry.market_label, edge_value)
        absolute_edge = abs(edge_value) if edge_value is not None else 0.0
        if absolute_edge < min_edge:
            continue

        board_rows.append(
            {
                "player_name": entry.player_name,
                "player_id": player_id,
                "market": entry.market_label,
                "opponent_abbr": opponent_abbr or "N/A",
                "game_date": game_date or "N/A",
                "sportsbook_line": round(entry.line_score, 1),
                "model_projection": model_projection,
                "edge": edge_value,
                "edge_label": edge_label,
                "edge_class": _edge_class(edge_value),
                "signal_label": signal_label,
                "signal_class": signal_class,
                "signal_note": signal_note,
                "absolute_edge": absolute_edge,
            }
        )

    board_rows.sort(key=lambda row: float(row["absolute_edge"]), reverse=True)
    pagination = _paginate_board_rows(board_rows, page=page)
    display_rows = pagination["rows"]
    message = "Biggest PrizePicks model edges ranked by absolute gap."
    if not board_entries:
        message = "PrizePicks returned no NBA board rows right now."
    elif not board_rows:
        message = "No rows met your current edge filter."

    return _board_context(
        title="PrizePicks Edge Board",
        source_name="PrizePicks",
        message=message,
        rows=display_rows,
        min_edge=min_edge,
        total_lines=len(board_entries),
        total_players=int(pagination["total_players"]),
        displayed_players=int(pagination["displayed_players"]),
        matched_players=len(prediction_cache),
        unmatched_players=len(unmatched_players),
        page=int(pagination["page"]),
        total_pages=int(pagination["total_pages"]),
    )


_DISCREPANCY_HISTORY_PATH = Path("data/discrepancy_history.csv")
_DISCREPANCY_HISTORY_FIELDS = ["snapshot_date", "player_name", "market", "opponent_abbr", "ud_line", "pp_line", "diff", "bet"]
_snapshot_save_lock = Lock()

def _save_discrepancy_snapshot(discrepancies: list) -> None:
    if not discrepancies:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    with _snapshot_save_lock:
        existing_dates: set[str] = set()
        if _DISCREPANCY_HISTORY_PATH.exists():
            with open(_DISCREPANCY_HISTORY_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    existing_dates.add(row.get("snapshot_date", ""))
        if today in existing_dates:
            return
        write_header = not _DISCREPANCY_HISTORY_PATH.exists()
        _DISCREPANCY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DISCREPANCY_HISTORY_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_DISCREPANCY_HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            for d in discrepancies:
                writer.writerow({"snapshot_date": today, **{k: d[k] for k in _DISCREPANCY_HISTORY_FIELDS[1:]}})


@app.route('/line-discrepancy/history')
def line_discrepancy_history():
    if not _DISCREPANCY_HISTORY_PATH.exists():
        return Response("No history yet.", status=404)
    with open(_DISCREPANCY_HISTORY_PATH, newline="") as f:
        content = f.read()
    return Response(
        content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=discrepancy_history.csv"},
    )


@app.route('/line-discrepancy')
def line_discrepancy():
    min_diff = max(_parse_optional_float(request.args.get("min_diff")) or 0.5, 0.0)

    pp_entries, ud_entries = [], []
    pp_error = ud_error = None

    try:
        pp_entries = prizepicks_client.fetch_board_entries()
    except Exception as exc:
        pp_error = str(exc)

    try:
        ud_entries = underdog_client.fetch_board_entries()
    except Exception as exc:
        ud_error = str(exc)

    def _norm_name(v: str) -> str:
        return " ".join(v.lower().replace(".", "").split())

    def _norm_market(v: str) -> str:
        return v.strip().lower()

    # Build PrizePicks lookup: (norm_name, norm_market) -> entry
    pp_lookup: dict[tuple[str, str], object] = {}
    for e in pp_entries:
        key = (_norm_name(e.player_name), _norm_market(e.market_label))
        pp_lookup[key] = e

    discrepancies = []
    for ud in ud_entries:
        key = (_norm_name(ud.player_name), _norm_market(ud.market_label))
        pp = pp_lookup.get(key)
        if pp is None:
            continue
        diff = ud.line_score - pp.line_score
        abs_diff = abs(diff)
        if abs_diff < min_diff:
            continue
        # Underdog lower → bet OVER (easier bar); Underdog higher → bet UNDER (easier bar)
        bet = "OVER" if ud.line_score < pp.line_score else "UNDER"
        discrepancies.append({
            "player_name": ud.player_name,
            "market": ud.market_label,
            "opponent_abbr": ud.opponent_abbr or pp.opponent_abbr or "N/A",
            "ud_line": round(ud.line_score, 1),
            "pp_line": round(pp.line_score, 1),
            "diff": round(abs_diff, 1),
            "bet": bet,
        })

    discrepancies.sort(key=lambda r: -r["diff"])

    # Auto-save snapshot once per day
    _save_discrepancy_snapshot(discrepancies)

    # Build parlay groups: top picks by diff for each size
    UNDERDOG_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 40.0}
    parlay_groups = []
    for size, payout in UNDERDOG_PAYOUTS.items():
        picks = discrepancies[:size]
        if len(picks) < size:
            continue
        parlay_groups.append({
            "size": size,
            "payout": payout,
            "picks": picks,
        })

    return render_template(
        "line_discrepancy.html",
        discrepancies=discrepancies,
        parlay_groups=parlay_groups,
        min_diff=min_diff,
        pp_error=pp_error,
        ud_error=ud_error,
        total_discrepancies=len(discrepancies),
    )


@app.route('/line-discrepancy/export')
def line_discrepancy_export():
    min_diff = max(_parse_optional_float(request.args.get("min_diff")) or 0.5, 0.0)

    pp_entries, ud_entries = [], []
    try:
        pp_entries = prizepicks_client.fetch_board_entries()
    except Exception:
        pass
    try:
        ud_entries = underdog_client.fetch_board_entries()
    except Exception:
        pass

    def _norm_name(v: str) -> str:
        return " ".join(v.lower().replace(".", "").split())

    def _norm_market(v: str) -> str:
        return v.strip().lower()

    pp_lookup = {}
    for e in pp_entries:
        pp_lookup[(_norm_name(e.player_name), _norm_market(e.market_label))] = e

    discrepancies = []
    for ud in ud_entries:
        key = (_norm_name(ud.player_name), _norm_market(ud.market_label))
        pp = pp_lookup.get(key)
        if pp is None:
            continue
        diff = ud.line_score - pp.line_score
        abs_diff = abs(diff)
        if abs_diff < min_diff:
            continue
        bet = "OVER" if ud.line_score < pp.line_score else "UNDER"
        discrepancies.append({
            "player_name": ud.player_name,
            "market": ud.market_label,
            "opponent_abbr": ud.opponent_abbr or pp.opponent_abbr or "",
            "underdog_line": round(ud.line_score, 1),
            "prizepicks_line": round(pp.line_score, 1),
            "gap": round(abs_diff, 1),
            "bet": bet,
        })

    discrepancies.sort(key=lambda r: -r["gap"])

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["player_name", "market", "opponent_abbr", "underdog_line", "prizepicks_line", "gap", "bet"])
    writer.writeheader()
    writer.writerows(discrepancies)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=underdog_value_plays.csv"},
    )


@app.route('/parlayplay-board')
def parlayplay_board():
    page = max(_parse_optional_int(request.args.get("page")) or 1, 1)
    min_edge = max(_parse_optional_float(request.args.get("min_edge")) or 0.0, 0.0)

    try:
        board_entries = parlayplay_client.fetch_board_entries()
    except ParlayPlayProviderError as exc:
        return _board_context(
            title="ParlayPlay Edge Board",
            source_name="ParlayPlay",
            message=str(exc),
            rows=[],
            min_edge=min_edge,
            total_lines=0,
            total_players=0,
            displayed_players=0,
            matched_players=0,
            unmatched_players=0,
            page=1,
            total_pages=1,
        )
    except Exception as exc:
        print(f"ParlayPlay board error: {exc}")
        return _board_context(
            title="ParlayPlay Edge Board",
            source_name="ParlayPlay",
            message="ParlayPlay board is unavailable right now.",
            rows=[],
            min_edge=min_edge,
            total_lines=0,
            total_players=0,
            displayed_players=0,
            matched_players=0,
            unmatched_players=0,
            page=1,
            total_pages=1,
        )

    prediction_cache: dict[tuple[str, str, str], dict[str, float]] = {}
    unmatched_players: set[str] = set()
    board_rows: list[dict[str, str | float | None]] = []

    for entry in board_entries:
        player_id = client.resolve_player_id_by_name(entry.player_name)
        if player_id is None:
            unmatched_players.add(entry.player_name)
            continue

        game_date = _parse_start_date(entry.start_time)
        opponent_abbr = entry.opponent_abbr if entry.opponent_abbr in NBA_TEAM_OPTIONS else ""
        cache_key = (str(player_id), opponent_abbr, game_date)
        if cache_key not in prediction_cache:
            try:
                points, assists, rebounds = predict_player_stats(
                    str(player_id),
                    opponent_abbr=opponent_abbr or None,
                    game_date=game_date or None,
                )
                prediction_cache[cache_key] = _build_prediction_summary(points, assists, rebounds)
            except Exception as exc:
                print(f"ParlayPlay board prediction error for {entry.player_name}: {exc}")
                continue

        prediction = prediction_cache[cache_key]
        model_projection = prediction[entry.market_label]
        edge_value, edge_label = _pick_edge_summary(
            model_projection,
            entry.line_score,
            selection_key=entry.selection_key,
            selection_label=entry.selection_label,
        )
        signal_label, signal_class, signal_note = _edge_signal(
            entry.market_label,
            edge_value,
            favor_positive_only=bool(entry.selection_key),
        )
        absolute_edge = abs(edge_value) if edge_value is not None else 0.0
        if absolute_edge < min_edge:
            continue

        board_rows.append(
            {
                "player_name": entry.player_name,
                "player_id": player_id,
                "market": entry.market_label,
                "selection_label": entry.selection_label or "Line",
                "payout_multiplier_label": f"{entry.payout_multiplier:.2f}x" if entry.payout_multiplier is not None else "N/A",
                "opponent_abbr": opponent_abbr or "N/A",
                "game_date": game_date or "N/A",
                "sportsbook_line": round(entry.line_score, 1),
                "model_projection": model_projection,
                "edge": edge_value,
                "edge_label": edge_label,
                "edge_class": _edge_class(edge_value),
                "signal_label": signal_label,
                "signal_class": signal_class,
                "signal_note": signal_note,
                "absolute_edge": absolute_edge,
            }
        )

    board_rows.sort(key=lambda row: float(row["absolute_edge"]), reverse=True)
    pagination = _paginate_board_rows(board_rows, page=page)
    display_rows = pagination["rows"]
    message = "Biggest ParlayPlay ladder edges ranked by pick strength."
    if not board_entries:
        message = "ParlayPlay returned no NBA board rows right now."
    elif not board_rows:
        message = "No rows met your current edge filter."

    return _board_context(
        title="ParlayPlay Edge Board",
        source_name="ParlayPlay",
        message=message,
        rows=display_rows,
        min_edge=min_edge,
        total_lines=len(board_entries),
        total_players=int(pagination["total_players"]),
        displayed_players=int(pagination["displayed_players"]),
        matched_players=len(prediction_cache),
        unmatched_players=len(unmatched_players),
        page=int(pagination["page"]),
        total_pages=int(pagination["total_pages"]),
    )


@app.route('/underdog-board')
def underdog_board():
    page = max(_parse_optional_int(request.args.get("page")) or 1, 1)
    min_edge = max(_parse_optional_float(request.args.get("min_edge")) or 0.0, 0.0)
    requested_market = request.args.get("market", "all").strip()
    selected_market = requested_market if requested_market in UNDERDOG_MARKET_FILTERS else "all"

    try:
        snapshot = _cached_underdog_board_snapshot()
    except UnderdogProviderError as exc:
        return _board_context(
            title="Underdog Edge Board",
            source_name="Underdog",
            message=str(exc),
            rows=[],
            min_edge=min_edge,
            total_lines=0,
            total_players=0,
            displayed_players=0,
            matched_players=0,
            unmatched_players=0,
            page=1,
            total_pages=1,
            market_filters=UNDERDOG_MARKET_FILTERS,
            selected_market=selected_market,
        )
    except Exception as exc:
        print(f"Underdog board error: {exc}")
        return _board_context(
            title="Underdog Edge Board",
            source_name="Underdog",
            message="Underdog board is unavailable right now.",
            rows=[],
            min_edge=min_edge,
            total_lines=0,
            total_players=0,
            displayed_players=0,
            matched_players=0,
            unmatched_players=0,
            page=1,
            total_pages=1,
            market_filters=UNDERDOG_MARKET_FILTERS,
            selected_market=selected_market,
        )

    board_rows = [
        dict(row)
        for row in snapshot["board_rows"]
        if float(row["absolute_edge"]) >= min_edge
    ]

    if selected_market != "all":
        board_rows = [row for row in board_rows if row["market"] == selected_market]

    parlay_combos = _build_parlay_combos(board_rows)

    snapshots = _load_snapshots()
    line_movers = _get_line_movers(board_rows, snapshots[-1]) if snapshots else []
    latest_snapshot_at = snapshots[-1]["saved_at"] if snapshots else None

    pagination = _paginate_board_rows(board_rows, page=page)
    display_rows = pagination["rows"]

    # Enrich with line shopping data
    def _norm(s: str) -> str:
        return " ".join(str(s).lower().replace(".", "").split())

    line_shop_map = _build_line_shopping_map()
    _ODDS_API_BOOKS = {"DraftKings", "FanDuel", "BetMGM", "BetRivers", "William Hill (US)"}
    enriched_rows = []
    for row in display_rows:
        r = dict(row)
        shop = line_shop_map.get((_norm(str(r.get("player_name", ""))), str(r.get("market", ""))), {})
        r["pp_line"] = shop.get("pp")
        r["pplay_line"] = shop.get("pplay")
        r["odds_api_lines"] = {k: v for k, v in shop.items() if k not in ("pp", "pplay") and v is not None}
        ud_line = float(r.get("sportsbook_line") or 0)
        proj = float(r.get("model_projection") or 0)
        best_line = ud_line
        best_book = "UD"
        all_lines = [("PP", r["pp_line"]), ("PPlay", r["pplay_line"])] + list(r["odds_api_lines"].items())
        for book, line in all_lines:
            if line is not None:
                if abs(proj - line) > abs(proj - best_line):
                    best_line = line
                    best_book = book
        r["best_line"] = best_line
        r["best_book"] = best_book
        r["best_edge"] = round(proj - best_line, 2) if proj else None
        enriched_rows.append(r)

    message = "Biggest Underdog pick edges ranked by pick strength."
    if not int(snapshot["total_lines"]):
        message = "Underdog returned no NBA board rows right now."
    elif not board_rows:
        if selected_market != "all":
            message = f"No {selected_market.lower()} rows met your current edge filter."
        else:
            message = "No rows met your current edge filter."
    elif selected_market != "all":
        message = f"Underdog {selected_market.lower()} props ranked by model edge."

    # Build cross-book discrepancies from all board rows (not just current page)
    all_enriched = []
    for row in board_rows:
        r = dict(row)
        shop = line_shop_map.get((_norm(str(r.get("player_name", ""))), str(r.get("market", ""))), {})
        r["pp_line"] = shop.get("pp")
        r["pplay_line"] = shop.get("pplay")
        all_enriched.append(r)

    book_discrepancies = []
    for r in all_enriched:
        ud = float(r.get("sportsbook_line") or 0)
        proj = float(r.get("model_projection") or 0)
        for book_key, book_label in [("pp_line", "PP"), ("pplay_line", "PPlay")]:
            other = r.get(book_key)
            if other is not None:
                gap = round(abs(float(other) - ud), 1)
                if gap >= 0.5:
                    book_discrepancies.append({
                        "player_name": r["player_name"],
                        "market": r["market"],
                        "ud_line": ud,
                        "other_line": float(other),
                        "other_book": book_label,
                        "gap": gap,
                        "model_projection": proj,
                        "ud_edge": round(proj - ud, 1) if proj else None,
                        "other_edge": round(proj - float(other), 1) if proj else None,
                        "signal_class": r.get("signal_class", ""),
                        "signal_label": r.get("signal_label", ""),
                        "player_id": r.get("player_id", ""),
                        "opponent_abbr": r.get("opponent_abbr", ""),
                        "game_date": r.get("game_date", ""),
                    })
    book_discrepancies.sort(key=lambda x: x["gap"], reverse=True)
    book_discrepancies = book_discrepancies[:20]

    return _board_context(
        title="Underdog Edge Board",
        source_name="Underdog",
        message=message,
        rows=enriched_rows,
        min_edge=min_edge,
        total_lines=int(snapshot["total_lines"]),
        total_players=int(pagination["total_players"]),
        displayed_players=int(pagination["displayed_players"]),
        matched_players=int(snapshot["matched_players"]),
        unmatched_players=int(snapshot["unmatched_players"]),
        page=int(pagination["page"]),
        total_pages=int(pagination["total_pages"]),
        market_filters=UNDERDOG_MARKET_FILTERS,
        selected_market=selected_market,
        parlay_combos=parlay_combos,
        line_movers=line_movers,
        latest_snapshot_at=latest_snapshot_at,
        book_discrepancies=book_discrepancies,
    )


def _build_line_shopping_map() -> dict[tuple[str, str], dict[str, float | None]]:
    """Fetch all books and return {(norm_name, market): {book_key: line}} lookup."""
    def _norm(s: str) -> str:
        return " ".join(str(s).lower().replace(".", "").split())

    result: dict[tuple[str, str], dict[str, float | None]] = {}

    try:
        for entry in prizepicks_client.fetch_board_entries():
            key = (_norm(entry.player_name), entry.market_label)
            result.setdefault(key, {})["pp"] = entry.line_score
    except Exception:
        pass

    try:
        for entry in parlayplay_client.fetch_board_entries():
            key = (_norm(entry.player_name), entry.market_label)
            result.setdefault(key, {})["pplay"] = entry.line_score
    except Exception:
        pass

    try:
        for (norm_name, market), book_lines in odds_api_client.build_line_map().items():
            key = (norm_name, market)
            result.setdefault(key, {}).update(book_lines)
    except (OddsApiProviderError, Exception):
        pass

    return result


_SNAPSHOT_FILE = settings.tracking_file.parent / "underdog_line_snapshots.json"
_MAX_SNAPSHOTS = 20


def _load_snapshots() -> list[dict]:
    try:
        if _SNAPSHOT_FILE.exists():
            import json
            return json.loads(_SNAPSHOT_FILE.read_text())
    except Exception:
        pass
    return []


def _save_snapshot(board_rows: list[dict]) -> str:
    import json
    snapshots = _load_snapshots()
    lines = {
        f"{row['player_name']}|{row['market']}": {
            "line": float(row["sportsbook_line"]),
            "model_projection": float(row["model_projection"] or 0),
            "edge": float(row["edge"] or 0),
            "selection_label": row.get("selection_label", ""),
            "opponent_abbr": row.get("opponent_abbr", ""),
        }
        for row in board_rows
    }
    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshots.append({"saved_at": saved_at, "lines": lines})
    snapshots = snapshots[-_MAX_SNAPSHOTS:]
    _SNAPSHOT_FILE.write_text(json.dumps(snapshots, indent=2))
    return saved_at


def _get_line_movers(current_rows: list[dict], snapshot: dict) -> list[dict]:
    snap_lines = snapshot.get("lines", {})
    saved_at = snapshot.get("saved_at", "")
    movers = []
    for row in current_rows:
        key = f"{row['player_name']}|{row['market']}"
        prev = snap_lines.get(key)
        if prev is None:
            continue
        delta = round(float(row["sportsbook_line"]) - prev["line"], 1)
        if delta == 0:
            continue
        edge_before = float(prev["edge"])
        edge_after = float(row["edge"] or 0)
        # Did the move help or hurt the pick?
        # Positive edge = model is OVER the line. If line goes up, edge shrinks.
        if edge_before > 0 and edge_after > edge_before:
            move_signal = "Confirms Pick"
            move_signal_class = "signal-lean"
        elif edge_before > 0 and edge_after <= 0:
            move_signal = "Line Flipped"
            move_signal_class = "edge-under"
        elif edge_before > 0:
            move_signal = "Erodes Edge"
            move_signal_class = "edge-neutral"
        else:
            move_signal = "Watch"
            move_signal_class = "edge-neutral"
        movers.append({
            "player_name": row["player_name"],
            "market": row["market"],
            "selection_label": row.get("selection_label", ""),
            "opponent_abbr": row.get("opponent_abbr", "N/A"),
            "line_before": prev["line"],
            "line_after": float(row["sportsbook_line"]),
            "delta": delta,
            "delta_label": f"+{delta}" if delta > 0 else str(delta),
            "edge_before": round(edge_before, 2),
            "edge_after": round(edge_after, 2),
            "move_signal": move_signal,
            "move_signal_class": move_signal_class,
            "saved_at": saved_at,
            "abs_delta": abs(delta),
        })
    movers.sort(key=lambda m: m["abs_delta"], reverse=True)
    return movers


@app.route('/live-scores')
def live_scores():
    try:
        from espn_game_client import get_scoreboard
        games = get_scoreboard()
    except Exception as exc:
        print(f"Live scores error: {exc}")
        games = []
    any_live = any(g["status"] == "live" for g in games)
    return render_template('live_scores.html', games=games, any_live=any_live)


@app.route('/live-scores/<game_id>')
def live_game_box_score(game_id):
    try:
        from espn_game_client import get_game_box_score
        box = get_game_box_score(game_id)
    except Exception as exc:
        print(f"Box score error: {exc}")
        box = {"game": None, "teams": []}
    game = box.get("game") or {}
    is_live = game.get("status") == "live"
    return render_template('game_box_score.html', box=box, game=game, is_live=is_live)


@app.route('/underdog-board/snapshot', methods=['POST'])
def save_underdog_snapshot():
    min_edge = max(_parse_optional_float(request.args.get("min_edge")) or 0.0, 0.0)
    selected_market = request.args.get("market", "all").strip()
    try:
        snapshot = _cached_underdog_board_snapshot()
        board_rows = [
            dict(row)
            for row in snapshot["board_rows"]
            if float(row["absolute_edge"]) >= min_edge
        ]
        if selected_market != "all":
            board_rows = [r for r in board_rows if r["market"] == selected_market]
        _save_snapshot(board_rows)
    except Exception as exc:
        print(f"Snapshot save error: {exc}")
    from flask import redirect, url_for
    return redirect(url_for('underdog_board', min_edge=min_edge, market=selected_market))


@app.route('/underdog-board/export')
def export_underdog_board():
    min_edge = max(_parse_optional_float(request.args.get("min_edge")) or 0.0, 0.0)
    requested_market = request.args.get("market", "all").strip()
    selected_market = requested_market if requested_market in UNDERDOG_MARKET_FILTERS else "all"

    try:
        snapshot = _cached_underdog_board_snapshot()
    except UnderdogProviderError as exc:
        return render_template('error.html', message=str(exc)), 503
    except Exception as exc:
        print(f"Underdog board export error: {exc}")
        return render_template('error.html', message="Unable to export the Underdog board right now."), 503

    board_rows = [
        dict(row)
        for row in snapshot["board_rows"]
        if float(row["absolute_edge"]) >= min_edge
    ]
    if selected_market != "all":
        board_rows = [row for row in board_rows if row["market"] == selected_market]

    tracking_rows = _build_tracking_rows_from_board_rows(board_rows, sportsbook="Underdog")
    logged_rows = _write_tracking_rows(tracking_rows)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=TRACKING_FIELDNAMES)
    writer.writeheader()
    writer.writerows(tracking_rows)

    market_slug = re.sub(r"[^a-z0-9]+", "-", selected_market.lower()).strip("-") if selected_market != "all" else "all-props"
    filename = f"underdog-board-{market_slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["X-Exported-Rows"] = str(len(tracking_rows))
    response.headers["X-Logged-Rows"] = str(logged_rows)
    return response


@app.route('/underdog-board/log-all', methods=['POST'])
def log_all_underdog_lines():
    try:
        snapshot = _cached_underdog_board_snapshot()
    except Exception as exc:
        print(f"Log all underdog error: {exc}")
        return {"error": "Unable to fetch Underdog board."}, 503

    board_rows = [dict(row) for row in snapshot["board_rows"]]
    tracking_rows = _build_tracking_rows_from_board_rows(board_rows, sportsbook="Underdog")
    logged_rows = _write_tracking_rows(tracking_rows)
    return {"logged": logged_rows, "total": len(tracking_rows)}


@app.route('/underdog-board/ncaab')
def underdog_ncaab_board():
    page = max(_parse_optional_int(request.args.get("page")) or 1, 1)
    min_edge = max(_parse_optional_float(request.args.get("min_edge")) or 0.0, 0.0)
    requested_market = request.args.get("market", "all").strip()
    selected_market = requested_market if requested_market in UNDERDOG_MARKET_FILTERS else "all"

    try:
        snapshot = _cached_ncaab_board_snapshot()
    except UnderdogProviderError as exc:
        return _board_context(
            title="Underdog NCAAB Edge Board",
            source_name="Underdog NCAAB",
            message=str(exc),
            rows=[], min_edge=min_edge, total_lines=0, total_players=0,
            displayed_players=0, matched_players=0, unmatched_players=0,
            page=1, total_pages=1, market_filters=UNDERDOG_MARKET_FILTERS,
            selected_market=selected_market,
        )
    except Exception as exc:
        print(f"Underdog NCAAB board error: {exc}")
        return _board_context(
            title="Underdog NCAAB Edge Board",
            source_name="Underdog NCAAB",
            message="NCAAB board is unavailable right now.",
            rows=[], min_edge=min_edge, total_lines=0, total_players=0,
            displayed_players=0, matched_players=0, unmatched_players=0,
            page=1, total_pages=1, market_filters=UNDERDOG_MARKET_FILTERS,
            selected_market=selected_market,
        )

    board_rows = [
        dict(row)
        for row in snapshot["board_rows"]
        if float(row["absolute_edge"]) >= min_edge
    ]
    if selected_market != "all":
        board_rows = [row for row in board_rows if row["market"] == selected_market]

    pagination = _paginate_board_rows(board_rows, page=page)

    return _board_context(
        title="Underdog NCAAB Edge Board",
        source_name="Underdog NCAAB",
        message="",
        rows=pagination["rows"],
        min_edge=min_edge,
        total_lines=int(snapshot["total_lines"]),
        total_players=int(snapshot["matched_players"]),
        displayed_players=len(pagination["rows"]),
        matched_players=int(snapshot["matched_players"]),
        unmatched_players=int(snapshot["unmatched_players"]),
        page=pagination["page"],
        total_pages=pagination["total_pages"],
        market_filters=UNDERDOG_MARKET_FILTERS,
        selected_market=selected_market,
    )


@app.route('/matchup-predict', methods=['POST'])
def matchup_predict():
    away_team = request.form.get("away_team", "").strip().upper()
    home_team = request.form.get("home_team", "").strip().upper()
    game_date = request.form.get("game_date", "").strip()

    if not away_team or not home_team or away_team == home_team:
        return render_template('error.html', message="Choose two different teams for the upcoming game.")

    try:
        away_rotation_all = [p for p in client.get_team_rotation(away_team, limit=50) if p["projected_minutes"] >= 10]
        home_rotation_all = [p for p in client.get_team_rotation(home_team, limit=50) if p["projected_minutes"] >= 10]

        # Filter confirmed Out/Doubtful players using ESPN injury report
        try:
            from injury_client import get_out_player_names, get_player_status
            away_out = get_out_player_names(away_team)
            home_out = get_out_player_names(home_team)
            away_rotation = [p for p in away_rotation_all if p["name"].lower() not in away_out]
            home_rotation = [p for p in home_rotation_all if p["name"].lower() not in home_out]
            def _inj_status(name, team_abbr):
                return get_player_status(name, team_abbr)
        except Exception:
            away_rotation = away_rotation_all
            home_rotation = home_rotation_all
            def _inj_status(name, team_abbr):
                return None
        slate = []
        calibration_summary = _build_calibration_summary()
        available_line_sources: list[tuple[str, object]] = []
        try:
            if underdog_client.is_configured():
                underdog_client.fetch_board_entries()
                available_line_sources.append(("Underdog", underdog_client))
        except Exception as exc:
            print(f"Underdog slate board preload error: {exc}")

        # Capture form data before entering threads — request context is thread-local
        form_data = dict(request.form)

        all_players = [
            (player, "Away", away_team, home_team)
            for player in away_rotation
        ] + [
            (player, "Home", home_team, away_team)
            for player in home_rotation
        ]

        def _predict_slate_player(args):
            player, side, team_abbr, opponent_abbr = args
            manual_line_inputs = _extract_slate_line_inputs(player["id"], form_data)
            prediction = predict_player_statline(
                str(player["id"]),
                opponent_abbr=opponent_abbr,
                game_date=game_date or None,
                home=(side == "Home"),
            )
            prediction_summary = _build_prediction_summary(
                prediction["points"],
                prediction["assists"],
                prediction["rebounds"],
            )
            line_inputs, line_sources = _fetch_slate_lines(
                player_name=player["name"],
                opponent_abbr=opponent_abbr,
                game_date=game_date,
                line_sources=available_line_sources,
            )
            line_inputs = _merge_fetched_lines(manual_line_inputs, line_inputs)
            confidence_summary = _build_market_confidence_summary(prediction)
            market_cards = [
                card
                for card in _build_market_cards(
                    prediction_summary,
                    line_inputs,
                    calibration_summary=calibration_summary,
                    confidence_summary=confidence_summary,
                )
                if card["line"] is not None
            ]
            ranked_market_cards = sorted(
                market_cards,
                key=lambda card: abs(float(card["edge"] or 0.0)),
                reverse=True,
            )
            market_card_lookup = {str(card["market"]): card for card in market_cards}
            return {
                "player_id": player["id"],
                "name": player["name"],
                "team_abbr": team_abbr,
                "opponent_abbr": opponent_abbr,
                "side": side,
                "minutes": round(player["minutes"], 1),
                "recent_minutes": round(player["recent_minutes"], 1),
                "projected_minutes": round(float(prediction.get("expected_minutes", player["projected_minutes"])), 1),
                "baseline_projected_minutes": round(player["projected_minutes"], 1),
                "minutes_trend": round(player["minutes_trend"], 1),
                "games_played": player["games_played"],
                "predicted_points": round(prediction["points"], 1),
                "predicted_assists": round(prediction["assists"], 1),
                "predicted_rebounds": round(prediction["rebounds"], 1),
                "predicted_points_rebounds": round(prediction["points"] + prediction["rebounds"], 1),
                "predicted_points_assists": round(prediction["points"] + prediction["assists"], 1),
                "predicted_assists_rebounds": round(prediction["assists"] + prediction["rebounds"], 1),
                "usage_rate": round(player["usage_rate"] * 100, 1),
                "recent_usage_rate": round(player["recent_usage_rate"] * 100, 1),
                "role_label": player["role_label"],
                "confidence_summary": confidence_summary,
                "market_cards": market_cards,
                "market_card_lookup": market_card_lookup,
                "line_source": line_sources,
                "best_edge_card": ranked_market_cards[0] if ranked_market_cards else None,
                "manual_line_inputs": manual_line_inputs,
                "injury_status": _inj_status(player["name"], team_abbr),
            }

        with ThreadPoolExecutor(max_workers=UNDERDOG_BOARD_PREDICTION_WORKERS) as executor:
            futures = {executor.submit(_predict_slate_player, args): args for args in all_players}
            for future in as_completed(futures):
                player_args = futures[future]
                try:
                    slate.append(future.result())
                except Exception as exc:
                    print(f"Matchup slate prediction error for player {player_args[0]['id']}: {exc}")

        if not slate:
            return render_template('error.html', message="No matchup slate predictions were generated for that game.")

        slate.sort(
            key=lambda player: (
                player["predicted_points"] + player["predicted_assists"] + player["predicted_rebounds"]
            ),
            reverse=True,
        )
        slate_summary = {
            "total_props_matched": sum(len(player["market_cards"]) for player in slate),
            "market_counts": {
                market_label: sum(
                    1
                    for player in slate
                    if market_label in player["market_card_lookup"]
                )
                for market_label, _ in TRACKED_MARKETS
            },
        }
        matchup = {"away_team": away_team, "home_team": home_team, "game_date": game_date}
        return render_template(
            'matchup_predictions.html',
            slate=slate,
            matchup=matchup,
            slate_summary=slate_summary,
            tracked_markets=[market_label for market_label, _ in TRACKED_MARKETS],
        )
    except Exception as exc:
        print(f"Matchup prediction error: {exc}")
        return render_template('error.html', message="Unable to build the matchup slate right now.")

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query'].strip()
    try:
        players = client.search_players(query) if query else []
        return render_template(
            'results.html',
            players=players,
            team_options=NBA_TEAM_OPTIONS,
            search_query=query,
        )
    except Exception as exc:
        print(f"Search error: {exc}")
        return render_template('error.html', message="Unable to search players right now.")

@app.route('/predict/<player_id>', methods=['GET', 'POST'])
def predict(player_id):
    player = client.get_player_details(player_id)
    if player is None:
        return render_template('error.html', message="Player not found")

    try:
        form_values = request.form if request.method == "POST" else request.args
        opponent_abbr = form_values.get("opponent_abbr", "").strip().upper()
        game_date = form_values.get("game_date", "").strip()
        sportsbook, line_inputs = _extract_line_inputs(form_values)
        line_source_notice = ""
        player_name = f"{player.get('firstname', '')} {player.get('lastname', '')}".strip() or f"Player {player_id}"

        if request.method == "POST":
            if form_values.get("autofill_prizepicks"):
                try:
                    fetched_lines = prizepicks_client.fetch_player_lines(
                        player_name=player_name,
                        opponent_abbr=opponent_abbr or None,
                        game_date=game_date or None,
                    )
                    if fetched_lines:
                        line_inputs = _merge_fetched_lines(line_inputs, fetched_lines)
                        if not form_values.get("sportsbook", "").strip():
                            sportsbook = "PrizePicks"
                        line_source_notice = f"Auto-filled {len(fetched_lines)} PrizePicks lines."
                    else:
                        line_source_notice = "PrizePicks did not return matching lines for that player and matchup."
                except PrizePicksProviderError as exc:
                    line_source_notice = str(exc)
                except Exception as exc:
                    print(f"PrizePicks autofill error: {exc}")
                    line_source_notice = "PrizePicks lines were unavailable, so the app used your manual inputs only."
            elif form_values.get("autofill_parlayplay"):
                try:
                    fetched_lines = parlayplay_client.fetch_player_lines(
                        player_name=player_name,
                        opponent_abbr=opponent_abbr or None,
                        game_date=game_date or None,
                    )
                    if fetched_lines:
                        line_inputs = _merge_fetched_lines(line_inputs, fetched_lines)
                        if not form_values.get("sportsbook", "").strip():
                            sportsbook = "ParlayPlay"
                        line_source_notice = f"Auto-filled {len(fetched_lines)} ParlayPlay lines."
                    else:
                        line_source_notice = "ParlayPlay did not return matching lines for that player and matchup."
                except ParlayPlayProviderError as exc:
                    line_source_notice = str(exc)
                except Exception as exc:
                    print(f"ParlayPlay autofill error: {exc}")
                    line_source_notice = "ParlayPlay lines were unavailable, so the app used your manual inputs only."
            elif form_values.get("autofill_underdog"):
                try:
                    fetched_lines = underdog_client.fetch_player_lines(
                        player_name=player_name,
                        opponent_abbr=opponent_abbr or None,
                        game_date=game_date or None,
                    )
                    if fetched_lines:
                        line_inputs = _merge_fetched_lines(line_inputs, fetched_lines)
                        if not form_values.get("sportsbook", "").strip():
                            sportsbook = "Underdog"
                        line_source_notice = f"Auto-filled {len(fetched_lines)} Underdog lines."
                    else:
                        line_source_notice = "Underdog did not return matching lines for that player and matchup."
                except UnderdogProviderError as exc:
                    line_source_notice = str(exc)
                except Exception as exc:
                    print(f"Underdog autofill error: {exc}")
                    line_source_notice = "Underdog lines were unavailable, so the app used your manual inputs only."
        prediction_payload = predict_player_statline(
            player_id,
            opponent_abbr=opponent_abbr or None,
            game_date=game_date or None,
        )
        prediction = _build_prediction_summary(
            prediction_payload["points"],
            prediction_payload["assists"],
            prediction_payload["rebounds"],
        )
        calibration_summary = _build_calibration_summary()
        confidence_summary = _build_market_confidence_summary(prediction_payload)
        market_cards = _build_market_cards(
            prediction,
            line_inputs,
            calibration_summary=calibration_summary,
            confidence_summary=confidence_summary,
        )
        has_line_inputs = any(card["line"] is not None for card in market_cards)
        logged_rows = _append_tracking_rows(
            player_id=player_id,
            player_name=player_name,
            opponent_abbr=opponent_abbr,
            game_date=game_date,
            sportsbook=sportsbook,
            market_cards=market_cards,
        )
        matchup = {
            "opponent_abbr": opponent_abbr,
            "game_date": game_date,
        }
        return render_template(
            'prediction.html',
            player=player,
            prediction=prediction,
            prediction_payload=prediction_payload,
            matchup=matchup,
            market_cards=market_cards,
            sportsbook=sportsbook,
            has_line_inputs=has_line_inputs,
            line_source_notice=line_source_notice,
            calibration_summary=calibration_summary,
            tracking_summary={
                "logged_rows": logged_rows,
                "tracking_file": str(settings.tracking_file),
            },
        )
    except Exception as exc:
        print(f"Prediction error: {exc}")
        return render_template('error.html', message="Unable to generate a prediction for that player right now.")


@app.route('/ncaa', methods=['GET', 'POST'])
def ncaa_page():
    if request.method == 'GET':
        return render_template('ncaa.html', result=None, error=None, form={})
    player_name = request.form.get('player_name', '').strip()
    stat_type = request.form.get('stat_type', 'points').strip()
    stat_line_raw = request.form.get('stat_line', '').strip()
    opponent = request.form.get('opponent', '').strip() or None
    if not player_name:
        return render_template('ncaa.html', result=None, error='Player name is required.', form=request.form)
    try:
        stat_line = float(stat_line_raw)
    except (ValueError, TypeError):
        return render_template('ncaa.html', result=None, error='Stat line must be a number.', form=request.form)
    try:
        result = predict_ncaa_player_prop(
            player_name,
            stat_type=stat_type,
            stat_line=stat_line,
            opponent=opponent,
        )
        return render_template('ncaa.html', result=result, error=None, form=request.form)
    except ValueError as exc:
        return render_template('ncaa.html', result=None, error=str(exc), form=request.form)
    except Exception as exc:
        return render_template('ncaa.html', result=None, error='Unable to generate a prediction right now.', form=request.form)


@app.route('/ncaa/predict', methods=['GET', 'POST'])
def predict_ncaa():
    payload = request.get_json(silent=True) or {}
    if not payload:
        payload = request.form if request.method == "POST" else request.args

    player_name = str(payload.get("player_name", "")).strip()
    stat_type = str(payload.get("stat_type") or payload.get("market") or "points").strip()
    stat_line = _parse_optional_float(payload.get("stat_line") or payload.get("line"))
    opponent = str(payload.get("opponent") or payload.get("opponent_abbr") or "").strip()
    season = _parse_optional_int(payload.get("season"))

    if not player_name:
        return {"error": "player_name is required."}, 400
    if stat_line is None:
        return {"error": "stat_line is required."}, 400

    try:
        return predict_ncaa_player_prop(
            player_name,
            stat_type=stat_type,
            stat_line=stat_line,
            opponent=opponent or None,
            season=season,
        )
    except ValueError as exc:
        return {"error": str(exc)}, 400
    except Exception as exc:
        print(f"NCAA prediction error: {exc}")
        return {"error": "Unable to generate an NCAA prediction right now."}, 500

if __name__ == '__main__':
    _start_underdog_board_prewarm()
    app.run(debug=settings.flask_debug, port=settings.flask_port)
