from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from flask_cors import CORS
from config import Config
from services import api_nba
from prediction import predict_player_stats

cache = Cache()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    cache.init_app(app, config={
        "CACHE_TYPE": app.config.get("CACHE_TYPE", "SimpleCache"),
        "CACHE_REDIS_URL": app.config.get("CACHE_REDIS_URL", "")
    })

    # Allow your Vercel site to call the API (tighten origin later)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # (Optional) existing HTML routes if you already use templates
    @app.route("/")
    def home():
        return render_template("index.html")

    @app.post("/search")
    @cache.cached(timeout=60, query_string=True)
    def search_html():
        q = request.form.get("query","").strip()
        data = api_nba.search_players(q) if q else {"response":[]}
        return render_template("results.html", players=data.get("response", []))

    # ---- JSON endpoints used by the Vercel frontend ----
    @app.get("/api/players")
    @cache.cached(timeout=60, query_string=True)
    def api_players():
        q = request.args.get("q","").strip()
        if not q:
            return jsonify({"players": []})
        raw = api_nba.search_players(q).get("response", [])
        players = [{
            "id": p.get("id"),
            "firstname": p.get("firstname"),
            "lastname": p.get("lastname"),
            "team": (p.get("team") or {}).get("name"),
            "number": p.get("leagues",{}).get("standard",{}).get("jersey"),
        } for p in raw]
        return jsonify({"players": players})

    @app.get("/api/predict")
    @cache.cached(timeout=30, query_string=True)
    def api_predict():
        pid = request.args.get("player_id")
        if not pid:
            return jsonify({"error":"player_id required"}), 400
        pts, ast, reb = predict_player_stats(pid)
        pd = api_nba.player_details(pid).get("response", [{}])[0]
        player = {"id": pd.get("id"),
                  "name": f"{pd.get('firstname','')} {pd.get('lastname','')}".strip(),
                  "team": (pd.get("team") or {}).get("name")}
        return jsonify({"player": player, "prediction": {
            "points": round(pts,1), "assists": round(ast,1), "rebounds": round(reb,1)
        }})

    @app.get("/api/schedule")
    @cache.cached(timeout=20, query_string=True)
    def api_schedule():
        from datetime import date
        date_iso = request.args.get("date", date.today().isoformat())
        return jsonify(api_nba.daily_schedule(date_iso))
    # ----------------------------------------------------

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
