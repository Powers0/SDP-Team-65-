from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from src.config import SEQ_LEN, PT_DIR, LOC_DIR, SHARED_DIR, SERVING_TABLE_PATH, LOCATION_GMM_PATH
from src.artifacts import load_all
from src.serving_data import load_serving_table, get_window_with_fallback
from src.inference import predict_next

import numpy as np

# Helper: convert numpy types to Python types for JSON serialization
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

# Keep repertoire pitch codes consistent with training (e.g., FO -> FS)
PITCH_MERGE = {
    "FO": "FS",  # forkball -> splitter family
    "SF": "FS",  # split-finger -> splitter family (optional)
}

def norm_pitch_type(pt: str) -> str:
    pt = str(pt).upper().strip()
    return PITCH_MERGE.get(pt, pt)

def apply_repertoire_mask(probs: dict, pitcher_mlbam: int, repertoire_map: dict) -> dict:
    """Zero out pitch types not in the pitcher's observed repertoire, then renormalize.

    Notes:
    - We normalize pitch codes (FO/SF -> FS) on both sides so masking is consistent
      with training-time merges.
    - We return a NEW dict in all cases.
    """
    allowed_raw = repertoire_map.get(int(pitcher_mlbam))
    if not allowed_raw:
        return {str(k): float(v) for k, v in probs.items()}

    allowed = {norm_pitch_type(a) for a in allowed_raw}

    masked = {}
    for k, v in probs.items():
        k_str = str(k)
        k_norm = norm_pitch_type(k_str)
        masked[k_norm] = masked.get(k_norm, 0.0) + float(v) if k_norm in allowed else masked.get(k_norm, 0.0)

    # Ensure every model class key appears (stable keys for the frontend)
    # If the incoming probs already have stable keys, this keeps them stable.
    for k, v in probs.items():
        k_norm = norm_pitch_type(str(k))
        masked.setdefault(k_norm, 0.0)

    s = float(sum(masked.values()))
    if s <= 0:
        # Safety fallback: if we somehow masked everything, return original (normalized keys)
        return {norm_pitch_type(str(k)): float(v) for k, v in probs.items()}

    return {k: (v / s) for k, v in masked.items()}

def sample_from_probs(probs: dict) -> tuple[str, float]:
    """Return (pitch_type, prob) sampled from a {code: p} dict."""
    items = [(k, float(v)) for k, v in probs.items()]
    keys = [k for k, _ in items]
    p = np.array([v for _, v in items], dtype=float)
    p = p / p.sum()
    idx = int(np.random.choice(len(keys), p=p))
    return keys[idx], float(p[idx])



def build_repertoire_map(df, min_usage: float = 0.02):
    """Build a per-pitcher set of allowed pitch types.

    A pitch type must account for at least `min_usage` (default 2%) of a
    pitcher's total pitches to be included. This prevents historical outlier
    pitches (e.g. a sinker a pitcher threw 1% of the time in 2021 but abandoned)
    from leaking through the repertoire mask at inference time.
    """
    rep = {}
    for pid, group in df.groupby("pitcher"):
        # Normalize codes to match training merges (FO/SF -> FS)
        normalized = group["pitch_type"].dropna().map(norm_pitch_type).astype(str)
        counts = normalized.value_counts(normalize=True)
        types = set(counts[counts >= min_usage].index)
        rep[int(pid)] = types
    return rep

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": "http://localhost:5173"}},
)

print("Loading artifacts/models...")
ART = load_all(PT_DIR, LOC_DIR, SHARED_DIR, gmm_path=LOCATION_GMM_PATH)
print("PT_DIR:", os.path.abspath(PT_DIR))
print("Loaded pitchtype model input_shape:", ART["pitchtype_model"].input_shape)
print("pt_features len:", len(ART["pt_features"]))
print("pt_features last 5:", ART["pt_features"][-5:])
print("pt_scaler n_features_in_:", getattr(ART["pt_scaler_X"], "n_features_in_", None))

print("Loading serving table...")
SERVING_DF = load_serving_table(SERVING_TABLE_PATH)

REPERTOIRE_MAP = build_repertoire_map(SERVING_DF)

print("Built repertoire map for", len(REPERTOIRE_MAP), "pitchers")

ART["repertoire_map"] = REPERTOIRE_MAP

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
    # Optional user-selected context from frontend
    user_context = payload.get("user_context")

    window, context_label = get_window_with_fallback(
        SERVING_DF, pitcher_mlbam, batter_mlbam, SEQ_LEN
    )

    result = predict_next(
        window,
        ART,
        SEQ_LEN,
        pitcher_mlbam,
        batter_mlbam,
        user_context=user_context,
        sample_pitch_type=True,
    )

    
    
    out = to_py(result)
    out["context_label"] = context_label
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)