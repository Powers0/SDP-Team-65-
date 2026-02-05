import os
import json
import re
import pandas as pd
from pybaseball import playerid_reverse_lookup

CSV_DIR = "csv data"
YEARS = [2021, 2022, 2023, 2024]
OUT_PATH = os.path.join("artifacts", "shared", "player_names.json")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# -------------------------
# Name formatting 
# -------------------------

# Particles often lowercased inside last names (except if it's the FIRST token of the last name)
LOWER_PARTICLES = {"de", "del", "der", "la", "le", "van", "von", "da", "di", "du"}

# Full-name overrides for cases that cannot be inferred reliably from plain casing
# Key is normalized "first last" lowercased.
SPECIAL_FULLNAME_OVERRIDES = {
    "jacob degrom": "Jacob deGrom",
    # Add more as we notice them
}

def cap_piece(s: str) -> str:
    """Capitalize a simple chunk; handle McXxxx best-effort"""
    s = s.strip()
    if not s:
        return s

    w = s.lower()
    if w.startswith("mc") and len(w) > 2:
        return "Mc" + w[2:].capitalize()

    return w.capitalize()

def cap_token(token: str) -> str:
    """
    Capitalize a token that may contain hyphens/apostrophes:
    o'neill -> O'Neill
    smith-jones -> Smith-Jones
    """
    token = token.strip()
    if not token:
        return token

    parts = re.split(r"([\-'])", token.lower())  # keep separators
    out = []
    for p in parts:
        if p in {"-", "'"}:
            out.append(p)
        else:
            out.append(cap_piece(p))
    return "".join(out)

def format_person_name(first: str, last: str) -> str:
    """Format first/last name with practical rules + overrides."""
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
            last_fmt_tokens.append(tl)  # keep particle lowercase when not first in last name
        else:
            last_fmt_tokens.append(cap_token(t))
    last_fmt = " ".join(last_fmt_tokens)

    return f"{first_fmt} {last_fmt}".strip()

# -------------------------
# Build ID lists from statcast CSVs
# -------------------------

dfs = [
    pd.read_csv(os.path.join(CSV_DIR, f"statcast_full_{y}.csv"), usecols=["pitcher", "batter"])
    for y in YEARS
]
df = pd.concat(dfs, ignore_index=True).dropna(subset=["pitcher", "batter"])

pitcher_ids = sorted(df["pitcher"].astype(int).unique().tolist())
batter_ids  = sorted(df["batter"].astype(int).unique().tolist())

# -------------------------
# Lookup names + handedness from pybaseball
# -------------------------

pitcher_lookup = playerid_reverse_lookup(pitcher_ids, key_type="mlbam")
batter_lookup  = playerid_reverse_lookup(batter_ids, key_type="mlbam")

def to_map(lookup_df: pd.DataFrame) -> dict:
    """
    Output shape:
    {
      "<mlbam_id>": {
        "name": "First Last",
        "bats": "R" | "L" | "S" | None,
        "throws": "R" | "L" | None
      },
      ...
    }
    """
    m = {}

    
    has_bats = "bats" in lookup_df.columns
    has_throws = "throws" in lookup_df.columns

    for _, row in lookup_df.iterrows():
        mlbam = int(row["key_mlbam"])

        first = row.get("name_first", "")
        last = row.get("name_last", "")
        name = format_person_name(first, last)

        bats = row["bats"] if has_bats else None
        throws = row["throws"] if has_throws else None

        # Normalize NaN -> None
        if pd.isna(bats):
            bats = None
        if pd.isna(throws):
            throws = None

        # Normalize to single-letter strings if present
        if isinstance(bats, str):
            bats = bats.strip().upper()[:1]
        if isinstance(throws, str):
            throws = throws.strip().upper()[:1]

        m[str(mlbam)] = {
            "name": name,
            "bats": bats,
            "throws": throws,
        }

    return m

payload = {
    "pitchers": to_map(pitcher_lookup),
    "batters": to_map(batter_lookup),
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print("Wrote:", OUT_PATH)
print("Pitchers:", len(payload["pitchers"]), "Batters:", len(payload["batters"]))