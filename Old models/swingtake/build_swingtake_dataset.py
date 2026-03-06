import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

SEQ_LEN = 5
TEST_SIZE = 0.15
RANDOM_SEED = 42

import os

HERE = os.path.dirname(os.path.abspath(__file__))

CSV_DIR = os.path.join(HERE, "..", "..", "csv data")
ARTIFACTS = os.path.join(HERE, "artifacts")
ENCODER_DIR = os.path.join(HERE, "..", "artifacts")

TYPE_PROBS_PATH = os.path.join(HERE, "..", "..", "Pitch Type Prediction", "artifacts", "pitch_type_probs.npy")
PITCHLOC_ART_DIR = os.path.join(HERE, "..", "..", "Pitch Location Prediction", "artifacts")

YEARS = [2021, 2022, 2023, 2024]

# A reasonable Statcast swing definition 
SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}


def load_statcast():
    dfs = []
    for y in YEARS:
        path = os.path.join(CSV_DIR, f"statcast_full_{y}.csv")
        print("Loading:", path)
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def preprocess(df, pitcher_le, batter_le):
    features = [
        "balls", "strikes", "outs_when_up", "inning",
        "on_1b", "on_2b", "on_3b",
        "bat_score", "fld_score",
    ]

    # Need these for alignment + label
    needed = features + [
        "plate_x", "plate_z", "stand", "p_throws",
        "pitch_type", "pitcher", "batter",
        "game_date", "at_bat_number", "pitch_number",
        "description"
    ]

    df = df.dropna(subset=needed)

    for base in ["on_1b", "on_2b", "on_3b"]:
        df[base] = df[base].notna().astype(int)

    df["score_diff"] = df["bat_score"] - df["fld_score"]
    features.append("score_diff")

    # One-hot handedness
    df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)
    features += [c for c in df.columns if c.startswith("stand_") or c.startswith("p_throws_")]

    # Previous pitch location (matches location model's 16-feature scaler)
    df = df.sort_values(by=["pitcher", "game_date", "at_bat_number", "pitch_number"])
    df["prev_plate_x"] = df.groupby(["pitcher", "game_date", "at_bat_number"])["plate_x"].shift(1).fillna(0.0)
    df["prev_plate_z"] = df.groupby(["pitcher", "game_date", "at_bat_number"])["plate_z"].shift(1).fillna(0.0)
    features += ["prev_plate_x", "prev_plate_z"]

    # Encode pitcher/batter IDs using your shared encoders
    df["pitcher_id"] = pitcher_le.transform(df["pitcher"].astype(int))
    df["batter_id"] = batter_le.transform(df["batter"].astype(int))

    # Swing label
    df["swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)

    # (sorting handled above with prev_plate_x/z computation)

    X = df[features].astype(float).values
    swing = df["swing"].astype(int).values

    return df, X, swing, features


def build_sequences(df, X, swing, pitch_type_probs, scaler_X_pitchloc):
    
    seq_X = []
    seq_p = []
    seq_b = []
    seq_type = []
    y_swing = []

    grouped = df.groupby("pitcher").indices
    idx_pt = 0
    max_pt = pitch_type_probs.shape[0]

    for _, idxs in grouped.items():
        idxs = list(idxs)

        for i in range(len(idxs) - SEQ_LEN):
            if idx_pt >= max_pt:
                break

            win = idxs[i: i + SEQ_LEN]
            tgt = idxs[i + SEQ_LEN]

            seq_X.append(X[win])
            seq_p.append(df.iloc[tgt]["pitcher_id"])
            seq_b.append(df.iloc[tgt]["batter_id"])
            seq_type.append(pitch_type_probs[idx_pt])
            y_swing.append(swing[tgt])

            idx_pt += 1

    X_seq = np.array(seq_X)             # (N, seq_len, num_features_unscaled)
    P_seq = np.array(seq_p, dtype="int32")             # (N,)
    B_seq = np.array(seq_b, dtype="int32")             # (N,)
    PT_seq = np.array(seq_type)         # (N, pitch_type_dim)
    y = np.array(y_swing).astype(int)   # (N,)

    # Scale X using the same scaler for location model 
    ns, sl, nf = X_seq.shape
    X_seq_scaled = scaler_X_pitchloc.transform(X_seq.reshape(-1, nf)).reshape(ns, sl, nf)

    return X_seq_scaled, P_seq, B_seq, PT_seq, y


if __name__ == "__main__":
    os.makedirs(ARTIFACTS, exist_ok=True)

    pitcher_le = pickle.load(open(os.path.join(ENCODER_DIR, "pitcher_le.pkl"), "rb"))
    batter_le = pickle.load(open(os.path.join(ENCODER_DIR, "batter_le.pkl"), "rb"))

    # Load pitch-location scaler_X so the location model can consume our X
    scaler_X_pitchloc = pickle.load(open(os.path.join(PITCHLOC_ART_DIR, "scaler_X.pkl"), "rb"))

    print("Loading statcast data...")
    df = load_statcast()

    print("Preprocessing...")
    df, X, swing, features = preprocess(df, pitcher_le, batter_le)

    print("Loading pitch-type probabilities...")
    pitch_probs = np.load(TYPE_PROBS_PATH)

    print("Building sequences for Swing/Take...")
    X_seq, P_seq, B_seq, PT_seq, y = build_sequences(df, X, swing, pitch_probs, scaler_X_pitchloc)

    print("Train/test split...")
    X_train, X_test, P_train, P_test, B_train, B_test, PT_train, PT_test, y_train, y_test = train_test_split(
        X_seq, P_seq, B_seq, PT_seq, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print("Saving artifacts...")
    np.save(os.path.join(ARTIFACTS, "X_train.npy"), X_train)
    np.save(os.path.join(ARTIFACTS, "X_test.npy"), X_test)
    np.save(os.path.join(ARTIFACTS, "P_train.npy"), P_train)
    np.save(os.path.join(ARTIFACTS, "P_test.npy"), P_test)
    np.save(os.path.join(ARTIFACTS, "B_train.npy"), B_train)
    np.save(os.path.join(ARTIFACTS, "B_test.npy"), B_test)
    np.save(os.path.join(ARTIFACTS, "PT_train.npy"), PT_train)
    np.save(os.path.join(ARTIFACTS, "PT_test.npy"), PT_test)
    np.save(os.path.join(ARTIFACTS, "y_train.npy"), y_train)
    np.save(os.path.join(ARTIFACTS, "y_test.npy"), y_test)

    pickle.dump(features, open(os.path.join(ARTIFACTS, "features.pkl"), "wb"))

    print("\n✓ Swing/Take dataset built.")
    print("X_train:", X_train.shape, "PT_train:", PT_train.shape, "y_train:", y_train.shape)
