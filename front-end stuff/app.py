from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from pybaseball import playerid_reverse_lookup
from sklearn.preprocessing import LabelEncoder  # only if you want to reuse locally

# ----------------------------------------------------
# Config
# ----------------------------------------------------
SEQ_LEN = 5
PT_DIR = "../Pitch Type Prediction/artifacts/"
LOC_DIR = "../Pitch Location Prediction/artifacts/"
CSV_DIR = "../csv data/"

app = Flask(__name__)

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------
pitchtype_model = load_model(PT_DIR + "pitchtype_model.keras")
location_model  = load_model(LOC_DIR + "pitch_location_model.keras")

# ----------------------------------------------------
# Load Artifacts
# ----------------------------------------------------
pt_features  = pickle.load(open(PT_DIR + "features.pkl", "rb"))
pt_scaler_X  = pickle.load(open(PT_DIR + "scaler.pkl", "rb"))
pt_label_enc = pickle.load(open(PT_DIR + "label_encoder.pkl", "rb"))

loc_features = pickle.load(open(LOC_DIR + "features.pkl", "rb"))
loc_scaler_X = pickle.load(open(LOC_DIR + "scaler_X.pkl", "rb"))
loc_scaler_Y = pickle.load(open(LOC_DIR + "scaler_Y.pkl", "rb"))

# ----------------------------------------------------
# Load Statcast Data
# ----------------------------------------------------
dfs = []
for year in [2021, 2022, 2023, 2024]:
    dfs.append(pd.read_csv(f"{CSV_DIR}/statcast_full_{year}.csv"))

df = pd.concat(dfs, ignore_index=True)

# ----------------------------------------------------
# Build Names 
# ----------------------------------------------------
batter_ids = df["batter"].dropna().unique().astype(int)
pitcher_ids = df["pitcher"].dropna().unique().astype(int)

batter_lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")
pitcher_lookup = playerid_reverse_lookup(pitcher_ids, key_type="mlbam")

batter_map = {
    row["key_mlbam"]: f"{row['name_first']} {row['name_last']}"
    for _, row in batter_lookup.iterrows()
}
pitcher_map = {
    row["key_mlbam"]: f"{row['name_first']} {row['name_last']}"
    for _, row in pitcher_lookup.iterrows()
}

df["batter_name"] = df["batter"].map(batter_map).fillna("Unknown")
df["pitcher_name"] = df["pitcher"].map(pitcher_map).fillna("Unknown")

# ----------------------------------------------------
# Embedding IDs 
# ----------------------------------------------------
df["pitcher_embed"] = df["pitcher"].astype("category").cat.codes
df["batter_embed"]  = df["batter"].astype("category").cat.codes

# ----------------------------------------------------
# Pitch-type style feature engineering 
# ----------------------------------------------------
# Start from a copy so we don't accidentally break something if needed later
df = df.sort_values(["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"])

# Base numeric features
base_features = [
    "balls", "strikes", "outs_when_up", "inning",
    "on_1b", "on_2b", "on_3b",
    "bat_score", "fld_score"
]

target_col = "pitch_type"

# Base runners 0/1
for base in ["on_1b", "on_2b", "on_3b"]:
    df[base] = df[base].notna().astype(int)

# Fill handedness
df["stand"]    = df["stand"].fillna("R")
df["p_throws"] = df["p_throws"].fillna("R")

# Score diff
df["score_diff"] = df["bat_score"] - df["fld_score"]
base_features.append("score_diff")

# We don't drop rows by pitch_type / allowed set here,
# because we just need the *input features* for last SEQ_LEN pitches.
# The pitch_type model itself was trained on the filtered subset, but it
# can still consume the same feature structure.

# Fix zone missing (for previous_zone / consistency)
df["zone"] = df["zone"].fillna(-1)

# Fill numeric NaNs in base numeric features
for col in base_features:
    df[col] = df[col].fillna(df[col].median())

# One-hot encode handedness
df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)

# Previous pitch features (same logic as old app)
df["previous_pitch"] = df.groupby(["pitcher", "game_pk", "at_bat_number"])["pitch_type"].shift(1)
df["previous_zone"]  = df.groupby(["pitcher", "game_pk", "at_bat_number"])["zone"].shift(1)

df["previous_pitch"] = df["previous_pitch"].fillna("None")
df["previous_zone"]  = df["previous_zone"].fillna(-1)

df = pd.get_dummies(df, columns=["previous_pitch"], drop_first=False)


# ----------------------------------------------------
# Dropdown lists — NAMES ONLY (Option A)
# ----------------------------------------------------
HITTERS = sorted({name.title() for name in df["batter_name"].unique() if name != "Unknown"})
PITCHERS = sorted({name.title() for name in df["pitcher_name"].unique() if name != "Unknown"})

# Name → ID lookup dicts (title-cased keys to match dropdown)
name_to_batter_id = (
    df[["batter_name", "batter"]]
    .drop_duplicates()
    .assign(batter_name_title=lambda x: x["batter_name"].str.title())
    .set_index("batter_name_title")["batter"]
    .to_dict()
)

name_to_pitcher_id = (
    df[["pitcher_name", "pitcher"]]
    .drop_duplicates()
    .assign(pitcher_name_title=lambda x: x["pitcher_name"].str.title())
    .set_index("pitcher_name_title")["pitcher"]
    .to_dict()
)

# ----------------------------------------------------
# Utility: ensure all required feature columns exist
# ----------------------------------------------------
def ensure_columns(df_sub, required_cols):
    for col in required_cols:
        if col not in df_sub.columns:
            df_sub[col] = 0.0
    return df_sub[required_cols]

# ----------------------------------------------------
# Build Inputs
# ----------------------------------------------------
def build_sequence(batter_id, pitcher_id):
    # Filter to this matchup
    sub = df[(df["batter"] == batter_id) & (df["pitcher"] == pitcher_id)]

    # Need enough pitches
    if len(sub) < SEQ_LEN:
        raise ValueError("Not enough pitches in this matchup.")

    # Sort and take last SEQ_LEN as context
    sub = sub.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).tail(SEQ_LEN)

    # Make sure the feature matrices have all required columns
    sub_pt  = ensure_columns(sub.copy(), pt_features)
    sub_loc = ensure_columns(sub.copy(), loc_features)

    # Scale according to training scalers
    Xpt  = pt_scaler_X.transform(sub_pt.values)[np.newaxis, :, :]
    Xloc = loc_scaler_X.transform(sub_loc.values)[np.newaxis, :, :]

    p_ids = sub["pitcher_embed"].values.astype(int)[np.newaxis, :]
    b_ids = sub["batter_embed"].values.astype(int)[np.newaxis, :]

    return Xpt, Xloc, p_ids, b_ids

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    pitch_prediction = None
    location_prediction = None
    error = None

    if request.method == "POST":
        hitter_name = request.form["hitter"]
        pitcher_name = request.form["pitcher"]

        try:
            # Convert names → MLBAM IDs
            batter_id  = int(name_to_batter_id[hitter_name])
            pitcher_id = int(name_to_pitcher_id[pitcher_name])

            Xpt, Xloc, p_ids, b_ids = build_sequence(batter_id, pitcher_id)

            # 1) Pitch type prediction
            pt_probs = pitchtype_model.predict([Xpt, p_ids, b_ids], verbose=0)[0]
            pitch_prediction = pt_label_enc.classes_[np.argmax(pt_probs)]

            # 2) Location prediction
            loc_pred_scaled = location_model.predict(
                [Xloc, p_ids[:, 0], b_ids[:, 0], pt_probs[np.newaxis, :]],
                verbose=0
            )
            loc_pred = loc_scaler_Y.inverse_transform(loc_pred_scaled)[0]
            location_prediction = (
                round(loc_pred[0], 3),
                round(loc_pred[1], 3)
            )

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        hitters=HITTERS,
        pitchers=PITCHERS,
        prediction=pitch_prediction,
        location=location_prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)