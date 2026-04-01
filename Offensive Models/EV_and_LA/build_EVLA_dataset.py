import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEQ_LEN = 3
TEST_SIZE = 0.15
RANDOM_SEED = 42

CSV_DIR = "../../csv data"
ARTIFACTS = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"

YEARS = [2021, 2022, 2023, 2024]

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


def preprocess(df, pitcher_le, batter_le):
    # Context features used in the sliding window sequence
    seq_features = [
        "balls", "strikes", "outs_when_up", "inning",
        "bat_score", "fld_score",
    ]

    needed = seq_features + [
        "description", "pitch_type",
        "plate_x", "plate_z",
        "stand", "p_throws",
        "launch_speed", "launch_angle",
        "pitcher", "batter",
        "game_pk", "game_date", "at_bat_number", "pitch_number",
    ]

    df = df.dropna(subset=needed)
    df = df[df["pitch_type"].isin(ALL_PITCH_TYPES)].copy()

    for base in ["on_1b", "on_2b", "on_3b"]:
        df[base] = df[base].notna().astype(int)

    df["score_diff"] = df["bat_score"] - df["fld_score"]
    seq_features.append("score_diff")
    
    # Compute per-batter and per-pitcher mean LA from training data and merge it in
    df["batter_avg_la"] = df.groupby("batter")["launch_angle"].transform("mean")
    df["batter_avg_ev"] = df.groupby("batter")["launch_speed"].transform("mean")
    df["pitcher_avg_la"] = df.groupby("pitcher")["launch_angle"].transform("mean")
    df["pitcher_avg_ev"] = df.groupby("pitcher")["launch_speed"].transform("mean")
    seq_features += ["batter_avg_la", "batter_avg_ev", "pitcher_avg_la", "pitcher_avg_ev"]

    # One-hot handedness — same pattern as location model
    df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)
    seq_features += [c for c in df.columns if c.startswith("stand_") or c.startswith("p_throws_")]

    # Previous pitch location within the same plate appearance (no leakage)
    df = df.sort_values(by=["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"])
    grp = df.groupby(["pitcher", "game_pk", "at_bat_number"], sort=False)
    df["prev_plate_x"] = grp["plate_x"].shift(1).fillna(0.0)
    df["prev_plate_z"] = grp["plate_z"].shift(1).fillna(0.0)
    seq_features += ["prev_plate_x", "prev_plate_z"]

    # Encode pitcher/batter IDs
    df["pitcher_id"] = pitcher_le.transform(df["pitcher"].astype(int))
    df["batter_id"]  = batter_le.transform(df["batter"].astype(int))

    X = df[seq_features].astype(float).values

    return df, X, seq_features


def build_pitch_type_onehot(pitch_type: str) -> np.ndarray:
    """One-hot vector for a single pitch type string."""
    vec = np.zeros(len(ALL_PITCH_TYPES), dtype=np.float32)
    if pitch_type in ALL_PITCH_TYPES:
        vec[ALL_PITCH_TYPES.index(pitch_type)] = 1.0
    return vec


def build_location_features(plate_x: float, plate_z: float) -> np.ndarray:
    """
    Engineered location features for the TARGET pitch.
    Returns (4,): [plate_x, plate_z, dist_to_center, is_strike]
    """
    dist = np.sqrt(plate_x ** 2 + (plate_z - 2.5) ** 2)
    is_strike = float(abs(plate_x) <= 0.83 and 1.5 <= plate_z <= 3.5)
    return np.array([plate_x, plate_z, dist, is_strike], dtype=np.float32)


def build_sequences(df, X):
    seq_X, seq_p, seq_b, seq_pt, seq_loc, seq_y = [], [], [], [], [], []

    grouped = df.groupby(["pitcher", "game_pk", "at_bat_number"]).indices
    print(f"Total at-bats: {len(grouped):,}")
    print(f"Total pitches (post-filter): {len(df):,}")
    for _, idxs in grouped.items():
        idxs = list(idxs)

        for i, tgt_pos in enumerate(idxs):
            tgt_row = df.iloc[tgt_pos]

            if tgt_row["description"] not in BALLS_IN_PLAY:
                continue
            if pd.isna(tgt_row["launch_speed"]) or pd.isna(tgt_row["launch_angle"]):
                continue

            # Grab however many prior pitches exist, up to SEQ_LEN
            prior = idxs[max(0, i - SEQ_LEN):i]

            # Pad with zeros at the front if fewer than SEQ_LEN prior pitches
            window = np.zeros((SEQ_LEN, X.shape[1]), dtype=np.float32)
            if len(prior) > 0:
                window[SEQ_LEN - len(prior):] = X[prior]

            seq_X.append(window)
            seq_p.append(int(tgt_row["pitcher_id"]))
            seq_b.append(int(tgt_row["batter_id"]))
            seq_pt.append(build_pitch_type_onehot(str(tgt_row["pitch_type"])))
            seq_loc.append(build_location_features(
                float(tgt_row["plate_x"]), float(tgt_row["plate_z"])
            ))
            seq_y.append([float(tgt_row["launch_speed"]), float(tgt_row["launch_angle"])])

    X_seq  = np.array(seq_X,  dtype=np.float32)
    P_seq  = np.array(seq_p,  dtype=np.int32)
    B_seq  = np.array(seq_b,  dtype=np.int32)
    PT_seq = np.array(seq_pt, dtype=np.float32)
    LOC    = np.array(seq_loc, dtype=np.float32)
    y      = np.array(seq_y,  dtype=np.float32)

    return X_seq, P_seq, B_seq, PT_seq, LOC, y


if __name__ == "__main__":
    pitcher_le = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le  = pickle.load(open(SHARED_DIR + "batter_le.pkl", "rb"))

    print("Loading statcast data...")
    df = load_statcast()

    print("Preprocessing...")
    df, X, seq_features = preprocess(df, pitcher_le, batter_le)

    print("Building sequences...")
    X_seq, P_seq, B_seq, PT_seq, LOC, y = build_sequences(df, X)

    print(f"\nDataset size: {len(y):,} balls in play with full sequences")
    print(f"  EV  — mean: {y[:,0].mean():.1f}  std: {y[:,0].std():.1f}")
    print(f"  LA  — mean: {y[:,1].mean():.1f}  std: {y[:,1].std():.1f}")
    print(f"Shapes — X_seq: {X_seq.shape}  PT: {PT_seq.shape}  LOC: {LOC.shape}  y: {y.shape}")

    print("\nTrain/test split...")
    (X_train, X_test,
     P_train, P_test,
     B_train, B_test,
     PT_train, PT_test,
     LOC_train, LOC_test,
     y_train, y_test) = train_test_split(
        X_seq, P_seq, B_seq, PT_seq, LOC, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Scale the sequence features using train only (mirrors location model)
    ns_train, seq_len, nf = X_train.shape
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train.reshape(-1, nf)).reshape(ns_train, seq_len, nf)
    X_test_s  = scaler_X.transform(X_test.reshape(-1, nf)).reshape(X_test.shape[0], seq_len, nf)

    # Scale location features
    loc_scaler = StandardScaler()
    LOC_train_s = loc_scaler.fit_transform(LOC_train)
    LOC_test_s  = loc_scaler.transform(LOC_test)

    # Scale targets
    target_scaler = StandardScaler()
    y_train_s = target_scaler.fit_transform(y_train)
    y_test_s  = target_scaler.transform(y_test)

    print("Saving artifacts...")
    np.save(ARTIFACTS + "X_train.npy",   X_train_s)
    np.save(ARTIFACTS + "X_test.npy",    X_test_s)
    np.save(ARTIFACTS + "P_train.npy",   P_train)
    np.save(ARTIFACTS + "P_test.npy",    P_test)
    np.save(ARTIFACTS + "B_train.npy",   B_train)
    np.save(ARTIFACTS + "B_test.npy",    B_test)
    np.save(ARTIFACTS + "PT_train.npy",  PT_train)
    np.save(ARTIFACTS + "PT_test.npy",   PT_test)
    np.save(ARTIFACTS + "LOC_train.npy", LOC_train_s)
    np.save(ARTIFACTS + "LOC_test.npy",  LOC_test_s)
    np.save(ARTIFACTS + "y_train.npy",   y_train_s)
    np.save(ARTIFACTS + "y_test.npy",    y_test_s)

    pickle.dump(seq_features,   open(ARTIFACTS + "features.pkl",       "wb"))
    pickle.dump(scaler_X,       open(ARTIFACTS + "scaler_X.pkl",       "wb"))
    pickle.dump(loc_scaler,     open(ARTIFACTS + "loc_scaler.pkl",     "wb"))
    pickle.dump(target_scaler,  open(ARTIFACTS + "target_scaler.pkl",  "wb"))
    pickle.dump(ALL_PITCH_TYPES, open(ARTIFACTS + "pitch_types.pkl",   "wb"))

    print("\n✓ EV/LA sequence dataset built.")
    print(f"X_train: {X_train_s.shape}  PT_train: {PT_train.shape}  "
          f"LOC_train: {LOC_train_s.shape}  y_train: {y_train_s.shape}")

    print("P_seq min/max:", P_seq.min(), P_seq.max())
    print("B_seq min/max:", B_seq.min(), B_seq.max())