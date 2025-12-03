import pandas as pd
from pybaseball import playerid_reverse_lookup

# Load your full files exactly like the app
dfs = []
for y in [2021, 2022, 2023, 2024]:
    df = pd.read_csv(f"statcast_full_{y}.csv")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("Raw combined shape:", df.shape)

# Map names (just like app.py)
batter_lookup = playerid_reverse_lookup(df["batter"].dropna().unique().astype(int), key_type="mlbam")
pitcher_lookup = playerid_reverse_lookup(df["pitcher"].dropna().unique().astype(int), key_type="mlbam")

batter_id_to_name = {
    row["key_mlbam"]: (f"{row['name_first']} {row['name_last']}".lower())
    for _, row in batter_lookup.iterrows()
}

pitcher_id_to_name = {
    row["key_mlbam"]: (f"{row['name_first']} {row['name_last']}".lower())
    for _, row in pitcher_lookup.iterrows()
}

df["batter_name"] = df["batter"].map(batter_id_to_name).fillna("unknown")
df["pitcher_name"] = df["pitcher"].map(pitcher_id_to_name).fillna("unknown")

print("After name mapping:", df.shape)

# NOW TEST WITHOUT ANY DROPS
h = "pete alonso"
p = "aaron nola"

pair_raw = df[(df["batter_name"] == h) & (df["pitcher_name"] == p)]
print("\nRAW matchup rows (no drops):", len(pair_raw))

# Now apply EXACT SAME DROPS as app.py:
features = [
    "balls", "strikes", "outs_when_up", "inning",
    "on_1b", "on_2b", "on_3b", "bat_score", "fld_score"
]

df2 = df.dropna(subset=features + ["pitch_type", "stand", "p_throws", "zone"])
pair_after_drop = df2[(df2["batter_name"] == h) & (df2["pitcher_name"] == p)]

print("\nRows AFTER dropna:", len(pair_after_drop))

# If zero, print WHY they were dropped
if len(pair_after_drop) == 0:
    missing = df[df["batter_name"] == h]
    missing = missing[missing["pitcher_name"] == p]
    print("\nSample of missing rows (to inspect NaNs):")
    print(missing.head())