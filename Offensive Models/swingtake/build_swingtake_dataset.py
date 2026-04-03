import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.15
RANDOM_SEED = 42

CSV_DIR = "../../csv data"
ARTIFACTS = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
YEARS = [2021]#, 2022, 2023, 2024]

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

# All known Statcast pitch types — add/remove as needed
ALL_PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "KC", "ST", "SV", "CS", "FO", "KN", "EP"]


def load_statcast():
    dfs = []
    for y in YEARS:
        path = f"{CSV_DIR}/statcast_full_{y}.csv"
        print("Loading:", path)
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def preprocess(df):

    needed = [
        "plate_x", "plate_z", "stand", "p_throws",
        "pitch_type", "description",
        "balls", "strikes", "outs_when_up", "inning",
        "bat_score", "fld_score", "batter", "pitcher"
    ]

    df = df.dropna(subset=needed)

    # Drop unknown pitch types
    df = df[df["pitch_type"].isin(ALL_PITCH_TYPES)].copy()

    for base in ["on_1b", "on_2b", "on_3b"]:
        df[base] = df[base].notna().astype(int)
    
    df["score_diff"] = df["bat_score"] - df["fld_score"]


    

    # One-hot handedness
    df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)
    

    # Swing label
    df["swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)

    return df


def build_pitch_type_onehot(df):
    """One-hot encode actual pitch_type using a fixed vocab so dim is always consistent."""
    pt = pd.Categorical(df["pitch_type"], categories=ALL_PITCH_TYPES)
    onehot = pd.get_dummies(pt).values.astype(float)  # (N, len(ALL_PITCH_TYPES))
    return onehot


def build_location_features(df):
    """
    Build the same engineered location features as before, now from real plate_x / plate_z.
    Returns (N, 4): [plate_x, plate_z, dist_to_center, is_strike]
    """
    x = df["plate_x"].values
    z = df["plate_z"].values

    dist_to_center = np.sqrt((x - 0.0) ** 2 + (z - 2.5) ** 2)
    is_strike = ((np.abs(x) <= 0.83) & (z >= 1.5) & (z <= 3.5)).astype(float)

    return np.column_stack([x, z, dist_to_center, is_strike])

def build_context_features(df):
    stand_cols   = [c for c in df.columns if c.startswith("stand_")]
    pthrows_cols = [c for c in df.columns if c.startswith("p_throws_")]
    
    continuous_cols = ["balls", "strikes", "outs_when_up", "inning", "score_diff"]
    binary_cols = ["on_1b", "on_2b", "on_3b"] + stand_cols + pthrows_cols
    
    cols = continuous_cols + binary_cols
    return df[cols].values.astype(float), cols, len(continuous_cols)



if __name__ == "__main__":
    print("Loading statcast data...")
    df = load_statcast()

    print("Preprocessing...")
    df = preprocess(df)

    pitcher_le = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le  = pickle.load(open(SHARED_DIR + "batter_le.pkl", "rb"))

    df = df[df["pitcher"].astype(int).isin(pitcher_le.classes_)]
    df = df[df["batter"].astype(int).isin(batter_le.classes_)]

    PIT = pitcher_le.transform(df["pitcher"].astype(int)).reshape(-1, 1)
    BAT = batter_le.transform(df["batter"].astype(int)).reshape(-1, 1)


    print("Building pitch-type one-hots from actual Statcast pitch_type...")
    PT = build_pitch_type_onehot(df)  # (N, pitch_type_dim)

    print("Building location features from actual plate_x / plate_z...")
    LOC = build_location_features(df)  # (N, 4)
    CTX, ctx_feature_names, n_continuous = build_context_features(df)



    y = df["swing"].values.astype(int)

    print(f"Dataset size: {len(y)} pitches  |  swings: {y.mean():.3f}")

    print("Train/test split...")
    PT_train, PT_test, LOC_train, LOC_test, CTX_train, CTX_test, PIT_train, PIT_test, BAT_train, BAT_test, y_train, y_test = train_test_split(
        PT, LOC, CTX, PIT, BAT, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
)



    print("Saving artifacts...")
    np.save(ARTIFACTS + "PT_train.npy", PT_train)
    np.save(ARTIFACTS + "PT_test.npy",  PT_test)
    np.save(ARTIFACTS + "LOC_train.npy", LOC_train)
    np.save(ARTIFACTS + "LOC_test.npy",  LOC_test)
    np.save(ARTIFACTS + "y_train.npy",  y_train)
    np.save(ARTIFACTS + "y_test.npy",   y_test)
    np.save(ARTIFACTS + "CTX_train.npy", CTX_train)
    np.save(ARTIFACTS + "CTX_test.npy",  CTX_test)
    np.save(ARTIFACTS + "PIT_train.npy", PIT_train)
    np.save(ARTIFACTS + "PIT_test.npy",  PIT_test)
    np.save(ARTIFACTS + "BAT_train.npy", BAT_train)
    np.save(ARTIFACTS + "BAT_test.npy",  BAT_test)
    pickle.dump(n_continuous, open(ARTIFACTS + "ctx_n_continuous.pkl", "wb"))


    pickle.dump(ctx_feature_names, open(ARTIFACTS + "ctx_features.pkl", "wb"))


    pickle.dump(ALL_PITCH_TYPES, open(ARTIFACTS + "pitch_types.pkl", "wb"))

    print("\n✓ Swing/Take dataset built.")
    print("PT_train:", PT_train.shape, "LOC_train:", LOC_train.shape, "y_train:", y_train.shape)