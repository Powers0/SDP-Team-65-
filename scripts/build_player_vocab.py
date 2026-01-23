import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CSV_DIR = "../csv data"
OUT_DIR = "../artifacts/shared"
YEARS = [2021, 2022, 2023, 2024]

os.makedirs(OUT_DIR, exist_ok=True)

dfs = [pd.read_csv(f"{CSV_DIR}/statcast_full_{y}.csv", usecols=["pitcher", "batter"]) for y in YEARS]
df = pd.concat(dfs, ignore_index=True)

# Drop missing
df = df.dropna(subset=["pitcher", "batter"])

# Convert to int MLBAM IDs
pitchers = df["pitcher"].astype(int)
batters  = df["batter"].astype(int)

pitcher_le = LabelEncoder()
batter_le  = LabelEncoder()

pitcher_le.fit(pitchers)
batter_le.fit(batters)

with open(os.path.join(OUT_DIR, "pitcher_le.pkl"), "wb") as f:
    pickle.dump(pitcher_le, f)

with open(os.path.join(OUT_DIR, "batter_le.pkl"), "wb") as f:
    pickle.dump(batter_le, f)

print("Saved:")
print(" -", os.path.join(OUT_DIR, "pitcher_le.pkl"), "num_pitchers=", len(pitcher_le.classes_))
print(" -", os.path.join(OUT_DIR, "batter_le.pkl"), "num_batters=", len(batter_le.classes_))