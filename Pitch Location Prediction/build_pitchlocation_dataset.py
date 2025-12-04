# build_pitchlocation_dataset.py

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEQ_LEN = 8
TEST_SIZE = 0.15
RANDOM_SEED = 42

CSV_DIR = "../csv data/"
TYPE_PROBS_PATH = "../Pitch Type Prediction/artifacts/pitch_type_probs.npy"
ARTIFACTS = "artifacts/"

# If you want **exact** data behavior as the original code that used 2022–2024:
YEARS = [2021, 2022, 2023, 2024]
# If you truly want 2021 too, change to [2021, 2022, 2023, 2024]
# but then your results will obviously differ from the old 2022–24 script.


def load_statcast():
    dfs = []
    for y in YEARS:
        path = f"{CSV_DIR}/statcast_full_{y}.csv"
        print("Loading:", path)
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def preprocess(df):
    features = [
        "balls","strikes","outs_when_up","inning",
        "on_1b","on_2b","on_3b",
        "bat_score","fld_score"
    ]
    target_cols = ["plate_x", "plate_z"]

    df = df.dropna(subset=features + target_cols +
                   ["stand","p_throws","zone","pitch_type","pitcher","batter"])

    for base in ["on_1b","on_2b","on_3b"]:
        df[base] = df[base].notna().astype(int)

    df["score_diff"] = df["bat_score"] - df["fld_score"]
    features.append("score_diff")

    df = pd.get_dummies(df, columns=["stand","p_throws"], drop_first=False)
    features += [c for c in df.columns if c.startswith("stand_") or c.startswith("p_throws_")]

    df["pitcher_id"] = df["pitcher"].astype("category").cat.codes
    df["batter_id"] = df["batter"].astype("category").cat.codes

    df = df.sort_values(by=["pitcher","game_date","at_bat_number","pitch_number"])

    X = df[features].astype(float).values
    Y = df[["plate_x","plate_z"]].astype(float).values

    return df, X, Y, features


def build_sequences(df, X, Y, pitch_type_probs):
    seq_X, seq_Y = [], []
    seq_p, seq_b = [], []
    seq_type = []

    grouped = df.groupby("pitcher").indices
    idx_pt = 0
    max_pt = pitch_type_probs.shape[0]

    for _, idxs in grouped.items():
        idxs = list(idxs)

        for i in range(len(idxs) - SEQ_LEN):
            if idx_pt >= max_pt:
                break

            win = idxs[i : i + SEQ_LEN]
            tgt = idxs[i + SEQ_LEN]

            seq_X.append(X[win])
            seq_Y.append(Y[tgt])
            seq_p.append(df.iloc[tgt]["pitcher_id"])
            seq_b.append(df.iloc[tgt]["batter_id"])
            seq_type.append(pitch_type_probs[idx_pt])

            idx_pt += 1

    return (
        np.array(seq_X),
        np.array(seq_Y),
        np.array(seq_p),
        np.array(seq_b),
        np.array(seq_type)
    )


if __name__ == "__main__":
    print("Loading statcast data...")
    df = load_statcast()

    print("Preprocessing...")
    df, X, Y, features = preprocess(df)

    print("Loading pitch-type probabilities...")
    pitch_probs = np.load(TYPE_PROBS_PATH)

    print("Building sequences...")
    X_seq, Y_seq, P_seq, B_seq, PT_seq = build_sequences(df, X, Y, pitch_probs)

    print("Train/test split (exactly like original)...")
    (
        X_train, X_test,
        Y_train, Y_test,
        P_train, P_test,
        B_train, B_test,
        PT_train, PT_test
    ) = train_test_split(
        X_seq, Y_seq,
        P_seq, B_seq,
        PT_seq,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print("Scaling using ONLY training data...")
    ns_train, seq_len, nf = X_train.shape

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, nf)).reshape(ns_train, seq_len, nf)
    X_test_scaled  = scaler_X.transform(X_test.reshape(-1, nf)).reshape(X_test.shape[0], seq_len, nf)

    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled  = scaler_Y.transform(Y_test)

    print("Saving artifacts...")

    # Save train/test separately for everything
    np.save(ARTIFACTS + "X_train.npy", X_train_scaled)
    np.save(ARTIFACTS + "X_test.npy",  X_test_scaled)

    np.save(ARTIFACTS + "Y_train.npy", Y_train_scaled)
    np.save(ARTIFACTS + "Y_test.npy",  Y_test_scaled)

    np.save(ARTIFACTS + "P_train.npy", P_train)
    np.save(ARTIFACTS + "P_test.npy",  P_test)

    np.save(ARTIFACTS + "B_train.npy", B_train)
    np.save(ARTIFACTS + "B_test.npy",  B_test)

    np.save(ARTIFACTS + "PT_train.npy", PT_train)
    np.save(ARTIFACTS + "PT_test.npy",  PT_test)

    pickle.dump(features, open(ARTIFACTS + "features.pkl", "wb"))
    pickle.dump(scaler_X, open(ARTIFACTS + "scaler_X.pkl", "wb"))
    pickle.dump(scaler_Y, open(ARTIFACTS + "scaler_Y.pkl", "wb"))

    print("\n✓ Dataset built. Behavior now matches the original script (apart from YEARS choice).")

