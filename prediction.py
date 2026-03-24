from api_client import NBAApiClient
from live_context import build_upcoming_context
from modeling import load_predictor_bundle


def predict_player_statline(player_id, opponent_abbr=None, game_date=None, home=None):
    client = NBAApiClient()
    game_logs = client.get_player_statistics(player_id)
    if not game_logs:
        raise ValueError(f"No game logs found for player_id={player_id}.")

    bundle = load_predictor_bundle()
    upcoming_context = build_upcoming_context(
        game_logs,
        opponent_abbr=opponent_abbr,
        game_date=game_date,
        player_id=player_id,
        home=home,
    )
    predictions = bundle.predict(game_logs, upcoming_context=upcoming_context)

    # Surface teammate context for display
    predictions["teammate_availability"] = round(float(upcoming_context.get("teammate_availability", 1.0)), 3)
    predictions["minutes_opportunity_factor"] = round(float(upcoming_context.get("minutes_opportunity_factor", 1.0)), 3)

    # Surface game-day status
    predictions["game_status"] = upcoming_context.get("game_status", "unknown")
    predictions["game_status_detail"] = upcoming_context.get("game_status_detail", "")
    confirmed_starter_val = float(upcoming_context.get("confirmed_starter", -1.0))
    predictions["confirmed_starter"] = (
        True if confirmed_starter_val == 1.0
        else False if confirmed_starter_val == 0.0
        else None
    )
    # Warn when prediction may be unreliable
    warnings = []
    if predictions["game_status"] == "postponed":
        warnings.append("Game postponed — prediction may be invalid.")
    elif predictions["game_status"] == "live":
        warnings.append("Game is in progress — prediction reflects pre-game projection only.")
    elif predictions["game_status"] == "final":
        warnings.append("Game already completed.")
    if predictions["confirmed_starter"] is False:
        warnings.append("Player not in starting lineup — minutes projection may be lower than model expects.")
    predictions["game_day_warnings"] = warnings

    # When key teammates are newly missing, the model's rolling averages lag behind the opportunity shift.
    # Apply a conservative minutes-based scaling to bridge that gap.
    opportunity_factor = predictions["minutes_opportunity_factor"]
    if opportunity_factor > 1.02:
        baseline_minutes = predictions.get("expected_minutes", 30.0)
        # Apply 40% of the opportunity boost — conservative to avoid double-counting
        # when the rolling window has already partially captured the elevated role
        adjusted_minutes = min(42.0, baseline_minutes * (1.0 + (opportunity_factor - 1.0) * 0.4))
        minutes_scale = adjusted_minutes / max(baseline_minutes, 1.0)
        predictions["expected_minutes"] = round(adjusted_minutes, 1)
        for stat in ("points", "assists", "rebounds"):
            if stat in predictions:
                predictions[stat] = round(predictions[stat] * minutes_scale, 2)

    return predictions


def predict_player_stats(player_id, opponent_abbr=None, game_date=None, home=None):
    predictions = predict_player_statline(
        player_id,
        opponent_abbr=opponent_abbr,
        game_date=game_date,
        home=home,
    )
    return (
        predictions["points"],
        predictions["assists"],
        predictions["rebounds"],
    )
