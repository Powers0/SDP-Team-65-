from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import warnings
from pybaseball import cache, playerid_reverse_lookup
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# -------------------------------------------------
# Flask + Model
# -------------------------------------------------
app = Flask(__name__)
model = load_model("pitch_model.keras")

sequence_length = 5
cache.enable()
warnings.filterwarnings("ignore", category=FutureWarning)

print("\n=== Loading FULL Statcast data ===")

# -------------------------------------------------
# Load statcast_full_YYYY.csv files
# -------------------------------------------------
dfs = []
for year in [2021, 2022, 2023, 2024]:
    fname = f"statcast_full_{year}.csv"
    print("Loading:", fname)
    dfs.append(pd.read_csv(fname))

df = pd.concat(dfs, ignore_index=True)
print("Combined shape:", df.shape)

# -------------------------------------------------
# Add player names (batter_name / pitcher_name)
# -------------------------------------------------
print("Looking up player names...")

batter_ids = df["batter"].dropna().unique().astype(int)
pitcher_ids = df["pitcher"].dropna().unique().astype(int)

batter_lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")
pitcher_lookup = playerid_reverse_lookup(pitcher_ids, key_type="mlbam")

batter_id_to_name = {
    row["key_mlbam"]: f"{row['name_first']} {row['name_last']}".lower()
    for _, row in batter_lookup.iterrows()
}

pitcher_id_to_name = {
    row["key_mlbam"]: f"{row['name_first']} {row['name_last']}".lower()
    for _, row in pitcher_lookup.iterrows()
}

df["batter_name"] = df["batter"].map(batter_id_to_name).fillna("unknown")
df["pitcher_name"] = df["pitcher"].map(pitcher_id_to_name).fillna("unknown")

print("Name mapping complete.")

# -------------------------------------------------
# Preprocessing (Safe — no more deleting matchups)
# -------------------------------------------------

features = [
    "balls", "strikes", "outs_when_up", "inning",
    "on_1b", "on_2b", "on_3b",
    "bat_score", "fld_score"
]

target = "pitch_type"

# Base runners: already safe
for base in ["on_1b", "on_2b", "on_3b"]:
    df[base] = df[base].notna().astype(int)

# Handedness
df["stand"] = df["stand"].fillna("R")
df["p_throws"] = df["p_throws"].fillna("R")

# Score diff
df["score_diff"] = df["bat_score"] - df["fld_score"]
features.append("score_diff")

# Pitch type MUST exist
df = df.dropna(subset=[target])

# Allowed pitch types
common_pitches = ["FF", "SL", "SI", "CH", "CU", "FC", "ST"]
df = df[df["pitch_type"].isin(common_pitches)]

# Fix zone missing
df["zone"] = df["zone"].fillna(-1)

# Fill numeric NaNs with medians
for col in features:
    df[col] = df[col].fillna(df[col].median())

# One-hot encode handedness
df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)
features += [c for c in df.columns if c.startswith("stand_") or c.startswith("p_throws_")]

# Sort
df = df.sort_values(by=["pitcher", "game_date", "at_bat_number", "pitch_number"])

# Encode pitch_type labels
le_pitch = LabelEncoder()
le_pitch.fit(df["pitch_type"])

# Previous pitch features
df["previous_pitch"] = df.groupby(["pitcher", "game_pk", "at_bat_number"])["pitch_type"].shift(1)
df["previous_zone"]  = df.groupby(["pitcher", "game_pk", "at_bat_number"])["zone"].shift(1)

df["previous_pitch"] = df["previous_pitch"].fillna("None")
df["previous_zone"]  = df["previous_zone"].fillna(-1)

# One-hot previous pitch
df = pd.get_dummies(df, columns=["previous_pitch"], drop_first=False)
features += [c for c in df.columns if c.startswith("previous_pitch_")]
features.append("previous_zone")

# Normalize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Embedding IDs
df["pitcher_id_embed"] = df["pitcher"].astype("category").cat.codes
df["batter_id_embed"]  = df["batter"].astype("category").cat.codes

print("Preprocessing complete.")

# -------------------------------------------------
# Build 5-pitch sequence for a matchup
# -------------------------------------------------
def build_sequence_for_matchup(hitter_name, pitcher_name):
    hitter_name = hitter_name.lower()
    pitcher_name = pitcher_name.lower()

    pair_df = df[(df["batter_name"] == hitter_name) &
                 (df["pitcher_name"] == pitcher_name)]

    if pair_df.empty:
        raise ValueError(f"No matchup found for {pitcher_name.title()} vs {hitter_name.title()}")

    pair_df = pair_df.sort_values(by=["game_date", "game_pk", "at_bat_number", "pitch_number"])
    ab_ids = pair_df[["game_pk", "at_bat_number"]].drop_duplicates().values[::-1]

    chosen_ab = None
    for game_pk, ab_num in ab_ids:
        ab = pair_df[(pair_df["game_pk"] == game_pk) &
                     (pair_df["at_bat_number"] == ab_num)]
        if len(ab) >= sequence_length:
            chosen_ab = ab
            break

    if chosen_ab is None:
        raise ValueError("No AB with ≥ 5 pitches")

    context = chosen_ab.sort_values("pitch_number").iloc[-sequence_length:]

    X = context[features].values[np.newaxis, :, :].astype(np.float32)
    pitch_ids = context["pitcher_id_embed"].values[np.newaxis, :].astype(np.int32)
    batt_ids  = context["batter_id_embed"].values[np.newaxis, :].astype(np.int32)

    return X, pitch_ids, batt_ids

# -------------------------------------------------
# Dropdown lists
# -------------------------------------------------
HITTERS  = sorted({n.title() for n in df["batter_name"].unique() if n != "unknown"})
PITCHERS = sorted({n.title() for n in df["pitcher_name"].unique() if n != "unknown"})

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    error_message = None

    if request.method == "POST":
        hitter = request.form["hitter"]
        pitcher = request.form["pitcher"]

        try:
            X, pitch_ids, batt_ids = build_sequence_for_matchup(hitter, pitcher)
            probs = model.predict([X, pitch_ids, batt_ids], verbose=0)[0]
            prediction_label = le_pitch.classes_[np.argmax(probs)]
        except Exception as e:
            error_message = str(e)

    return render_template(
        "index.html",
        hitters=HITTERS,
        pitchers=PITCHERS,
        prediction=prediction_label,
        error=error_message
    )

if __name__ == "__main__":
    app.run(debug=True)