import os, json
import pandas as pd
from pybaseball import playerid_reverse_lookup

CSV_DIR = "csv data"
YEARS = [2021, 2022, 2023, 2024]
OUT_PATH = os.path.join("artifacts", "shared", "player_names.json")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

dfs = [pd.read_csv(os.path.join(CSV_DIR, f"statcast_full_{y}.csv"), usecols=["pitcher", "batter"]) for y in YEARS]
df = pd.concat(dfs, ignore_index=True).dropna(subset=["pitcher", "batter"])

pitcher_ids = sorted(df["pitcher"].astype(int).unique().tolist())
batter_ids  = sorted(df["batter"].astype(int).unique().tolist())

pitcher_lookup = playerid_reverse_lookup(pitcher_ids, key_type="mlbam")
batter_lookup  = playerid_reverse_lookup(batter_ids, key_type="mlbam")

def to_map(lookup_df):
    m = {}
    for _, row in lookup_df.iterrows():
        mlbam = int(row["key_mlbam"])
        name = f"{row['name_first']} {row['name_last']}".strip()
        m[str(mlbam)] = name
    return m

payload = {
    "pitchers": to_map(pitcher_lookup),
    "batters": to_map(batter_lookup),
}

with open(OUT_PATH, "w") as f:
    json.dump(payload, f, indent=2)

print("Wrote:", OUT_PATH)
print("Pitchers:", len(payload["pitchers"]), "Batters:", len(payload["batters"]))