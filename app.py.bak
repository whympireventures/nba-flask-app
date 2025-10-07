from flask import Flask, render_template, request, jsonify
import requests
from prediction import predict_player_stats

app = Flask(__name__)

def search_players(query):
    url = "https://api-nba-v1.p.rapidapi.com/players"
    querystring = {"search": query}
    headers = {
        "x-rapidapi-key": "4cbcf8a971msh3985c29941e0d6fp1d4320jsn0fd9ac1a026e",
        "x-rapidapi-host": "api-nba-v1.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return {"response": []}

def get_player_details(player_id):
    url = "https://api-nba-v1.p.rapidapi.com/players"
    querystring = {"id": str(player_id)}
    headers = {
        "x-rapidapi-key": "4cbcf8a971msh3985c29941e0d6fp1d4320jsn0fd9ac1a026e",
        "x-rapidapi-host": "api-nba-v1.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        if data.get('response'):
            return data['response'][0]
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    players = search_players(query)
    return render_template('results.html', players=players['response'])

@app.route('/send_player_id', methods=['POST'])
def send_player_id():
    data = request.get_json()
    player_id = data.get('player_id')
    print(f"Received player ID: {player_id}")
    return jsonify({"status": "success", "player_id": player_id})

@app.route('/predict/<player_id>')
def predict(player_id):
    print(f"Predicting stats for player ID: {player_id}")
    player = get_player_details(player_id)
    if player is None:
        return render_template('error.html', message="Player not found")
    print(f"Player details: {player}")
    try:
        points, assists, rebounds = predict_player_stats(player_id)
        prediction = {
            "Points": points,
            "Assists": assists,
            "Rebounds": rebounds
        }
        return render_template('prediction.html', player=player, prediction=prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('error.html', message="Player not found")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
