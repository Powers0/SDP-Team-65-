
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


CSV_DIR   = "../csv data"
YEARS     = [2021, 2022, 2023, 2024]
OUT_PATH  = "artifacts/location_gmm.pkl"

# Number of GMM components per pitch type.
# 3-5; more components = richer shape but more data needed.
N_COMPONENTS_DEFAULT = 4

# Override per pitch type if needed
N_COMPONENTS_OVERRIDE = {
    "FF": 4,
    "FT": 4,
    "SI": 4,
    "FC": 4,
    "FS": 3,
    "SF": 3,
    "SL": 5,
    "ST": 5,
    "CU": 5,
    "KC": 4,
    "CH": 4,
    "KN": 3,
    "EP": 3,
}

# Normalise a few aliases so they match training-time merges.
PITCH_MERGE = {
    "FO": "FS",
    "SF": "FS",  # split-finger → splitter family (same merge as build_pitchlocation_dataset.py)
}

# Realistic physical bounds (feet) – same as inference.py
PLATE_X_MIN, PLATE_X_MAX = -2.5, 2.5
PLATE_Z_MIN, PLATE_Z_MAX = 0.5,  5.0

# Minimum number of pitches to fit a GMM for a given type.
MIN_PITCHES = 500


def load_statcast() -> pd.DataFrame:
    dfs = []
    for y in YEARS:
        path = os.path.join(CSV_DIR, f"statcast_full_{y}.csv")
        print(f"  Loading {path} …")
        dfs.append(pd.read_csv(path, usecols=["pitch_type", "plate_x", "plate_z"]))
    return pd.concat(dfs, ignore_index=True)


def normalise_pitch_type(s: str) -> str:
    s = str(s).upper().strip()
    return PITCH_MERGE.get(s, s)


def fit_gmms(df: pd.DataFrame) -> dict:
    """Return {pitch_type: GaussianMixture} for every type with enough data."""
    df = df.dropna(subset=["pitch_type", "plate_x", "plate_z"]).copy()

    # Filter to physical bounds (removes bad-tracking rows)
    df = df[
        (df["plate_x"].between(PLATE_X_MIN, PLATE_X_MAX)) &
        (df["plate_z"].between(PLATE_Z_MIN, PLATE_Z_MAX))
    ]

    df["pitch_type"] = df["pitch_type"].map(normalise_pitch_type)

    gmm_dict = {}
    for pt, group in df.groupby("pitch_type"):
        coords = group[["plate_x", "plate_z"]].values
        if len(coords) < MIN_PITCHES:
            print(f"  [{pt}] only {len(coords)} rows → skipping (< {MIN_PITCHES})")
            continue

        n = N_COMPONENTS_OVERRIDE.get(str(pt), N_COMPONENTS_DEFAULT)
        print(f"  [{pt}] {len(coords):,} pitches → fitting GMM(n_components={n}) …")

        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",   
            max_iter=200,
            n_init=5,                 
            random_state=42,
        )
        gmm.fit(coords)
        gmm_dict[str(pt)] = gmm
        print(f"     means:\n{gmm.means_}")

    return gmm_dict


def main():
    print("Loading Statcast data …")
    df = load_statcast()
    print(f"Total rows: {len(df):,}")

    print("\nFitting GMMs …")
    gmm_dict = fit_gmms(df)

    os.makedirs(os.path.dirname(OUT_PATH) if os.path.dirname(OUT_PATH) else ".", exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(gmm_dict, f)

    print(f"\nSaved {len(gmm_dict)} GMMs → {OUT_PATH}")
    print("Pitch types:", list(gmm_dict.keys()))


if __name__ == "__main__":
    main()
