from flask import Flask, request, jsonify
from flask_cors import CORS

from src.config import SEQ_LEN, PT_DIR, LOC_DIR, SHARED_DIR, SERVING_TABLE_PATH
from src.artifacts import load_all
from src.serving_data import load_serving_table, get_window_with_fallback
from src.inference import predict_next

import numpy as np

def to_py(x):
    # numpy scalars
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()

    # dict/list recursion
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]

    return x

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": "http://localhost:5173"}},
)

print("Loading artifacts/models...")
ART = load_all(PT_DIR, LOC_DIR, SHARED_DIR)

print("Loading serving table...")
SERVING_DF = load_serving_table(SERVING_TABLE_PATH)

@app.get("/api/players")
def api_players():
    names_p = ART["player_names"]["pitchers"]
    names_b = ART["player_names"]["batters"]

    # ---- tuning knobs ----
    MIN_PITCH_COUNT = 200   # "real pitcher" threshold
    MIN_PA_COUNT = 50       # "real batter" threshold
    OHTANI_ID = 660271      # Shohei Ohtani MLBAM
    # ----------------------

    def get_meta(meta_source, pid_str):
        meta = meta_source.get(pid_str)
        if isinstance(meta, dict):
            return meta
        # backward compat if it's a string
        if isinstance(meta, str):
            return {"name": meta}
        return {"name": pid_str}

    def is_real_pitcher(pid_str):
        mp = get_meta(names_p, pid_str)
        # if missing, treat as 0
        return (mp.get("pitch_count") or 0) >= MIN_PITCH_COUNT

    def is_real_batter(pid_str):
        mb = get_meta(names_b, pid_str)
        return (mb.get("pa_count") or 0) >= MIN_PA_COUNT

    pitchers = []
    for x in ART["pitcher_le"].classes_:
        pid = str(int(x))
        mp = get_meta(names_p, pid)

        # filter out "blowout cameo" pitchers, except Ohtani
        if int(pid) != OHTANI_ID and not is_real_pitcher(pid):
            continue

        pitchers.append({
            "id": int(pid),
            "label": mp.get("name", pid),
            "bats": mp.get("bats"),
            "throws": mp.get("throws"),
        })

    batters = []
    for x in ART["batter_le"].classes_:
        bid = str(int(x))
        mb = get_meta(names_b, bid)

        # filter out pitchers-as-batters (unless Ohtani)
        if int(bid) != OHTANI_ID:
            # require minimum PA to be a selectable batter
            if not is_real_batter(bid):
                continue
            # if they are a real pitcher, exclude them from batters
            if is_real_pitcher(bid):
                continue

        batters.append({
            "id": int(bid),
            "label": mb.get("name", bid),
            "bats": mb.get("bats"),
            "throws": mb.get("throws"),
            "sz_top": mb.get("sz_top"),
            "sz_bot": mb.get("sz_bot"),
        })

    return jsonify({"pitchers": pitchers, "batters": batters})

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True)
    pitcher_mlbam = int(payload["pitcher_mlbam"])
    batter_mlbam  = int(payload["batter_mlbam"])

    window, context_label = get_window_with_fallback(
        SERVING_DF, pitcher_mlbam, batter_mlbam, SEQ_LEN
    )

    result = predict_next(window, ART, SEQ_LEN, pitcher_mlbam, batter_mlbam)

    out = to_py(result)
    out["context_label"] = context_label
    

    return jsonify(out)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)