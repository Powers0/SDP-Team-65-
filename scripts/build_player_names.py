import os
import json
import re
import pandas as pd
from pybaseball import playerid_reverse_lookup

CSV_DIR = "csv data"
YEARS = [2021, 2022, 2023, 2024]
OUT_PATH = os.path.join("artifacts", "shared", "player_names.json")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

LOWER_PARTICLES = {"de", "del", "der", "la", "le", "van", "von", "da", "di", "du"}
SPECIAL_FULLNAME_OVERRIDES = {"jacob degrom": "Jacob deGrom"}

def cap_piece(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    w = s.lower()
    if w.startswith("mc") and len(w) > 2:
        return "Mc" + w[2:].capitalize()
    return w.capitalize()

def cap_token(token: str) -> str:
    token = token.strip()
    if not token:
        return token
    parts = re.split(r"([\-'])", token.lower())
    out = []
    for p in parts:
        if p in {"-", "'"}:
            out.append(p)
        else:
            out.append(cap_piece(p))
    return "".join(out)

def format_person_name(first: str, last: str) -> str:
    first = " ".join(str(first).split())
    last = " ".join(str(last).split())

    key = f"{first} {last}".strip().lower()
    if key in SPECIAL_FULLNAME_OVERRIDES:
        return SPECIAL_FULLNAME_OVERRIDES[key]

    first_fmt = " ".join(cap_token(t) for t in first.split())

    last_tokens = last.split()
    last_fmt_tokens = []
    for i, t in enumerate(last_tokens):
        tl = t.lower()
        if i > 0 and tl in LOWER_PARTICLES:
            last_fmt_tokens.append(tl)
        else:
            last_fmt_tokens.append(cap_token(t))
    last_fmt = " ".join(last_fmt_tokens)

    return f"{first_fmt} {last_fmt}".strip()

# -------------------------
# Load Statcast IDs + handedness + PA keys
# -------------------------
dfs = [
    pd.read_csv(
        os.path.join(CSV_DIR, f"statcast_full_{y}.csv"),
        usecols=["pitcher", "batter", "stand", "p_throws", "game_pk", "at_bat_number"],
    )
    for y in YEARS
]
df = pd.concat(dfs, ignore_index=True).dropna(subset=["pitcher", "batter"])

df["pitcher"] = df["pitcher"].astype(int)
df["batter"] = df["batter"].astype(int)

pitcher_ids = sorted(df["pitcher"].unique().tolist())
batter_ids  = sorted(df["batter"].unique().tolist())

# handedness maps
batter_bats_map = (
    df.dropna(subset=["stand"])
      .assign(stand=lambda d: d["stand"].astype(str).str.upper().str.strip())
      .groupby("batter")["stand"]
      .agg(lambda s: s.value_counts().index[0])
      .to_dict()
)

pitcher_throws_map = (
    df.dropna(subset=["p_throws"])
      .assign(p_throws=lambda d: d["p_throws"].astype(str).str.upper().str.strip())
      .groupby("pitcher")["p_throws"]
      .agg(lambda s: s.value_counts().index[0])
      .to_dict()
)

# role-count maps (for filtering)
pitch_count_map = df.groupby("pitcher").size().to_dict()

pa_count_map = (
    df.dropna(subset=["game_pk", "at_bat_number"])
      .groupby("batter")[["game_pk", "at_bat_number"]]
      .apply(lambda g: g.drop_duplicates().shape[0])
      .to_dict()
)

# -------------------------
# Lookup names from pybaseball
# -------------------------
pitcher_lookup = playerid_reverse_lookup(pitcher_ids, key_type="mlbam")
batter_lookup  = playerid_reverse_lookup(batter_ids, key_type="mlbam")

def to_map(ids, lookup_df: pd.DataFrame, bats_map=None, throws_map=None, pitch_ct=None, pa_ct=None) -> dict:
    bats_map = bats_map or {}
    throws_map = throws_map or {}
    pitch_ct = pitch_ct or {}
    pa_ct = pa_ct or {}

    name_rows = {}
    for _, row in lookup_df.iterrows():
        mlbam = int(row["key_mlbam"])
        name_rows[mlbam] = (row.get("name_first", ""), row.get("name_last", ""))

    m = {}
    for mlbam in ids:
        first, last = name_rows.get(mlbam, ("", ""))
        if (not str(first).strip()) and (not str(last).strip()):
            name = f"Unknown ({mlbam})"
        else:
            name = format_person_name(first, last)

        bats = bats_map.get(mlbam)
        throws = throws_map.get(mlbam)

        bats = bats.strip().upper()[:1] if isinstance(bats, str) else None
        throws = throws.strip().upper()[:1] if isinstance(throws, str) else None

        m[str(mlbam)] = {
            "name": name,
            "bats": bats,
            "throws": throws,
            "pitch_count": int(pitch_ct.get(mlbam, 0)),
            "pa_count": int(pa_ct.get(mlbam, 0)),
        }

    return m

payload = {
    "pitchers": to_map(
        pitcher_ids,
        pitcher_lookup,
        bats_map=None,
        throws_map=pitcher_throws_map,
        pitch_ct=pitch_count_map,
        pa_ct=pa_count_map,
    ),
    "batters": to_map(
        batter_ids,
        batter_lookup,
        bats_map=batter_bats_map,
        throws_map=None,
        pitch_ct=pitch_count_map,
        pa_ct=pa_count_map,
    ),
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print("Wrote:", OUT_PATH)
print("Pitchers:", len(payload["pitchers"]), "Batters:", len(payload["batters"]))