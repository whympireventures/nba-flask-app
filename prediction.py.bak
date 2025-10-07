import pickle
import requests
import pandas as pd
from lightgbm import LGBMRegressor

def predict_player_stats(player_id):
    url = "https://api-nba-v1.p.rapidapi.com/players/statistics"
    querystring = {"id":player_id,"season":"2023"}
    headers = {
        "X-RapidAPI-Key": "4cbcf8a971msh3985c29941e0d6fp1d4320jsn0fd9ac1a026e",
        "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        all_matches = []
        for game in response.json()['response']:
            all_matches.append(game['game']['id'])
        
        # Sort all matches in descending order and get last 5
        all_matches.sort(reverse=True)
        last_5_match_ids = all_matches[:5]
        
        # Initialize statistics variables
        stats = {
            'points': 0, 'fgm': 0, 'fga': 0, 'fgp': 0.0,
            'ftp': 0.0, 'tpm': 0, 'tpa': 0, 'tpp': 0.0,
            'offReb': 0, 'defReb': 0, 'totReb': 0,
            'assists': 0, 'pFouls': 0, 'steals': 0,
            'turnovers': 0, 'blocks': 0, 'plusMinus': 0
        }
        
        match_count = 0
        response_data = response.json()['response']
        
        # Aggregate statistics
        for game in response_data:
            if game['game']['id'] in last_5_match_ids:
                match_count += 1
                for stat in stats:
                    if stat in ['fgp', 'ftp', 'tpp']:
                        stats[stat] += float(game[stat])
                    else:
                        stats[stat] += int(game[stat])
        
        # Calculate averages
        if match_count > 0:
            data = {}
            for stat in stats:
                data[f'last_5_matches_{stat}'] = stats[stat] / match_count
            
            # Create DataFrame
            df_last_5_matches = pd.DataFrame([data])
            
            # Load models and predict
            with open('model_points.pkl', 'rb') as f:
                model_points = pickle.load(f)
            with open('model_assists.pkl', 'rb') as f:
                model_assists = pickle.load(f)
            with open('model_rebounds.pkl', 'rb') as f:
                model_rebounds = pickle.load(f)
                
            points = model_points.predict(df_last_5_matches)
            assists = model_assists.predict(df_last_5_matches)
            rebounds = model_rebounds.predict(df_last_5_matches)
            
            return float(points[0]), float(assists[0]), float(rebounds[0])
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 0, 0, 0