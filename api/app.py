from flask import Flask, request, jsonify
from flask_cors import CORS

from src.config import SEQ_LEN, PT_DIR, LOC_DIR, SHARED_DIR, SERVING_TABLE_PATH
from src.artifacts import load_all
from src.serving_data import load_serving_table, get_latest_window
from src.inference import predict_next

app = Flask(__name__)
CORS(app)  # allow React dev server to call the API

print("Loading artifacts/models...")
ART = load_all(PT_DIR, LOC_DIR, SHARED_DIR)

print("Loading serving table...")
SERVING_DF = load_serving_table(SERVING_TABLE_PATH)

@app.get("/api/players")
def api_players():
    names_p = ART["player_names"]["pitchers"]
    names_b = ART["player_names"]["batters"]

    pitchers = []
    for x in ART["pitcher_le"].classes_:
        pid = str(int(x))
        meta = names_p.get(pid)

        if isinstance(meta, dict):
            label = meta.get("name", pid)
            bats = meta.get("bats")
            throws = meta.get("throws")
        else:
            # backward-compat if meta is still a string
            label = meta if isinstance(meta, str) else pid
            bats = None
            throws = None

        pitchers.append({
            "id": int(pid),
            "label": label,     # string for react-select
            "bats": bats,
            "throws": throws,   # pitcher throwing hand is what you’ll likely display
        })

    batters = []
    for x in ART["batter_le"].classes_:
        bid = str(int(x))
        meta = names_b.get(bid)

        if isinstance(meta, dict):
            label = meta.get("name", bid)
            bats = meta.get("bats")      # batter batting hand is what you’ll likely display
            throws = meta.get("throws")
        else:
            label = meta if isinstance(meta, str) else bid
            bats = None
            throws = None

        batters.append({
            "id": int(bid),
            "label": label,    # string for react-select
            "bats": bats,
            "throws": throws,
        })

    return jsonify({"pitchers": pitchers, "batters": batters})

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True)
    pitcher_mlbam = int(payload["pitcher_mlbam"])
    batter_mlbam  = int(payload["batter_mlbam"])

    window = get_latest_window(SERVING_DF, pitcher_mlbam, batter_mlbam, SEQ_LEN)
    result = predict_next(window, ART, SEQ_LEN)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)