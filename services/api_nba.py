import requests
from flask import current_app

def _headers():
    return {
        "x-rapidapi-key": current_app.config["RAPIDAPI_KEY"],
        "x-rapidapi-host": current_app.config["RAPIDAPI_HOST"],
    }

def search_players(query):
    return requests.get(
        "https://api-nba-v1.p.rapidapi.com/players",
        headers=_headers(),
        params={"search": query}
    ).json()

def player_details(player_id):
    return requests.get(
        "https://api-nba-v1.p.rapidapi.com/players",
        headers=_headers(),
        params={"id": str(player_id)}
    ).json()

def player_game_stats(player_id, season):
    return requests.get(
        "https://api-nba-v1.p.rapidapi.com/players/statistics",
        headers=_headers(),
        params={"id": player_id, "season": season}
    ).json()

def daily_schedule(date_iso):
    return requests.get(
        "https://api-nba-v1.p.rapidapi.com/games",
        headers=_headers(),
        params={"date": date_iso}
    ).json()
