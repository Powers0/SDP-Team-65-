import os
import pickle
import pandas as pd

CSV_DIR = "csv data"
YEARS = [2021, 2022, 2023, 2024]

SHARED_DIR = "artifacts/shared"
OUT_DIR = "artifacts/serving"
OUT_PATH = os.path.join(OUT_DIR, "serving_table.parquet")

SEQ_ORDER_COLS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"]

# Load shared encoders (MLBAM -> embedding index)
with open(os.path.join(SHARED_DIR, "pitcher_le.pkl"), "rb") as f:
    pitcher_le = pickle.load(f)
with open(os.path.join(SHARED_DIR, "batter_le.pkl"), "rb") as f:
    batter_le = pickle.load(f)

os.makedirs(OUT_DIR, exist_ok=True)

dfs = []
for y in YEARS:
    path = os.path.join(CSV_DIR, f"statcast_full_{y}.csv")
    print("Loading:", path)
    dfs.append(pd.read_csv(path))
df = pd.concat(dfs, ignore_index=True)

# Minimal columns needed for existing feature engineering + sequencing
needed_cols = [
    "pitcher","batter","pitch_type","zone",
    "game_date","game_pk","at_bat_number","pitch_number",
    "balls","strikes","outs_when_up","inning",
    "on_1b","on_2b","on_3b",
    "bat_score","fld_score",
    "stand","p_throws",
    "plate_x","plate_z",
]
df = df[[c for c in needed_cols if c in df.columns]].copy()

# Drop rows missing key stuff for building features + sequence windows
df = df.dropna(subset=[
    "pitcher","batter","pitch_type","zone",
    "game_date","game_pk","at_bat_number","pitch_number",
    "balls","strikes","outs_when_up","inning",
    "bat_score","fld_score",
    "stand","p_throws",
])

# Base occupancy -> 0/1
for base in ["on_1b","on_2b","on_3b"]:
    if base in df.columns:
        df[base] = df[base].notna().astype(int)
    else:
        df[base] = 0

df["zone"] = df["zone"].fillna(-1)
df["stand"] = df["stand"].fillna("R")
df["p_throws"] = df["p_throws"].fillna("R")
df["score_diff"] = df["bat_score"] - df["fld_score"]

# Sort for “previous pitch” features
df = df.sort_values(SEQ_ORDER_COLS)

# Previous pitch/zone (within plate appearance)
df["previous_pitch"] = df.groupby(["pitcher","game_pk","at_bat_number"])["pitch_type"].shift(1).fillna("None")
df["previous_zone"]  = df.groupby(["pitcher","game_pk","at_bat_number"])["zone"].shift(1).fillna(-1)

# One-hot for handedness + previous_pitch
df = pd.get_dummies(df, columns=["stand","p_throws","previous_pitch"], drop_first=False)

# Stable embedding indices from shared encoders
df["pitcher_embed"] = pitcher_le.transform(df["pitcher"].astype(int))
df["batter_embed"]  = batter_le.transform(df["batter"].astype(int))

# Save
df.to_parquet(OUT_PATH, index=False)
print("Saved serving table:", OUT_PATH, "rows =", len(df))