import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.15
RANDOM_SEED = 42

CSV_DIR = "../../csv data"
ARTIFACTS = "artifacts/"

YEARS = [2021, 2022, 2023, 2024]

# Balls in play only 
BALLS_IN_PLAY = {
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

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
        "description", "pitch_type",
        "plate_x", "plate_z",
        "balls", "strikes",
        "stand", "p_throws",
        "outs_when_up", "inning",
        "on_1b", "on_2b", "on_3b",
        "bat_score", "fld_score",
        "launch_speed", "launch_angle",
    ]

    # Filter to balls in play first
    df = df[df["description"].isin(BALLS_IN_PLAY)].copy()

    df = df.dropna(subset=needed)

    # Drop unknown pitch types
    df = df[df["pitch_type"].isin(ALL_PITCH_TYPES)].copy()

    for base in ["on_1b", "on_2b", "on_3b"]:
        df[base] = df[base].notna().astype(int)

    df["score_diff"] = df["bat_score"] - df["fld_score"]

    print(f"Balls in play after filtering: {len(df):,}")
    print(f"  EV  — mean: {df['launch_speed'].mean():.1f}  std: {df['launch_speed'].std():.1f}  "
          f"min: {df['launch_speed'].min():.1f}  max: {df['launch_speed'].max():.1f}")
    print(f"  LA  — mean: {df['launch_angle'].mean():.1f}  std: {df['launch_angle'].std():.1f}  "
          f"min: {df['launch_angle'].min():.1f}  max: {df['launch_angle'].max():.1f}")

    return df


def build_pitch_type_onehot(df):
    """Shape: (N, len(ALL_PITCH_TYPES))"""
    pt = pd.Categorical(df["pitch_type"], categories=ALL_PITCH_TYPES)
    return pd.get_dummies(pt).values.astype(float)


def build_location_features(df):
    """
    Returns (N, 4): [plate_x, plate_z, dist_to_center, is_strike]
    """
    x = df["plate_x"].values
    z = df["plate_z"].values
    dist_to_center = np.sqrt(x ** 2 + (z - 2.5) ** 2)
    is_strike = ((np.abs(x) <= 0.83) & (z >= 1.5) & (z <= 3.5)).astype(float)
    return np.column_stack([x, z, dist_to_center, is_strike])


def build_context_features(df):
    """
    Count, handedness, and game features
    Returns (N, ctx_dim) and the list of feature names
    """
    ctx = df[["balls", "strikes", "outs_when_up", "inning", "score_diff",
              "on_1b", "on_2b", "on_3b"]].copy()

    # One-hot handedness
    stand_dummies  = pd.get_dummies(df["stand"],    prefix="stand")
    pthrows_dummies = pd.get_dummies(df["p_throws"], prefix="p_throws")
    ctx = pd.concat([ctx, stand_dummies, pthrows_dummies], axis=1)

    feature_names = list(ctx.columns)
    return ctx.values.astype(float), feature_names


if __name__ == "__main__":
    print("Loading statcast data...")
    df = load_statcast()

    print("Preprocessing...")
    df = preprocess(df)

    print("Building features...")
    PT  = build_pitch_type_onehot(df)           # (N, 14)
    LOC = build_location_features(df)            # (N, 4)
    CTX, ctx_feature_names = build_context_features(df)  # (N, ctx_dim)

    # Targets: EV and launch angle as a (N, 2) array
    y = df[["launch_speed", "launch_angle"]].values.astype(float)

    print(f"\nShapes — PT: {PT.shape}  LOC: {LOC.shape}  CTX: {CTX.shape}  y: {y.shape}")

    print("Train/test split...")
    (PT_train,  PT_test,
     LOC_train, LOC_test,
     CTX_train, CTX_test,
     y_train,   y_test) = train_test_split(
        PT, LOC, CTX, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print("Saving artifacts...")
    np.save(ARTIFACTS + "PT_train.npy",  PT_train)
    np.save(ARTIFACTS + "PT_test.npy",   PT_test)
    np.save(ARTIFACTS + "LOC_train.npy", LOC_train)
    np.save(ARTIFACTS + "LOC_test.npy",  LOC_test)
    np.save(ARTIFACTS + "CTX_train.npy", CTX_train)
    np.save(ARTIFACTS + "CTX_test.npy",  CTX_test)
    np.save(ARTIFACTS + "y_train.npy",   y_train)
    np.save(ARTIFACTS + "y_test.npy",    y_test)

    pickle.dump(ALL_PITCH_TYPES,    open(ARTIFACTS + "pitch_types.pkl",    "wb"))
    pickle.dump(ctx_feature_names,  open(ARTIFACTS + "ctx_features.pkl",   "wb"))

    print("\n✓ EV/LA dataset built.")
    print("PT_train:", PT_train.shape, " LOC_train:", LOC_train.shape,
          " CTX_train:", CTX_train.shape, " y_train:", y_train.shape)