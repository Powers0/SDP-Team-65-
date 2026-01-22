
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

SEQUENCE_LEN = 5

# Which raw CSVs to load
YEARS = [2021, 2022, 2023, 2024]


def load_statcast():
    dfs = []
    for y in YEARS:
        path = f"../csv data/statcast_full_{y}.csv"
        print("Loading:", path)
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def preprocess(df):
    # -------- features used for prediction -------- #
    features = [
        "balls", "strikes", "outs_when_up", "inning",
        "on_1b", "on_2b", "on_3b",
        "bat_score", "fld_score"
    ]

    common_pitches = ["FF", "SL", "SI", "CH", "CU", "FC", "ST"]

    df = df.dropna(subset=features + ["pitch_type", "stand", "p_throws", "zone"])
    df = df[df["pitch_type"].isin(common_pitches)]

    # Base runners = 0/1
    for base in ["on_1b", "on_2b", "on_3b"]:
        df[base] = df[base].notna().astype(int)

    df["score_diff"] = df["bat_score"] - df["fld_score"]
    features.append("score_diff")

    # Handedness one-hot
    df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)
    features += [c for c in df.columns if c.startswith("stand_") or c.startswith("p_throws_")]

    # Proper pitch ordering
    df = df.sort_values(by=["pitcher", "game_date", "at_bat_number", "pitch_number"])

    # Label encoder for pitch_type
    le_pitch = LabelEncoder()
    df["pitch_encoded"] = le_pitch.fit_transform(df["pitch_type"])

    # Previous pitch/zone
    df["previous_pitch"] = df.groupby(["pitcher", "game_pk", "at_bat_number"])["pitch_type"].shift(1).fillna("None")
    df["previous_zone"] = df.groupby(["pitcher", "game_pk", "at_bat_number"])["zone"].shift(1).fillna(-1)

    df = pd.get_dummies(df, columns=["previous_pitch"], drop_first=False)
    features += [c for c in df.columns if c.startswith("previous_pitch_")]
    features.append("previous_zone")

    # Scale features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Embedding IDs
    df["pitcher_id"] = df["pitcher"].astype("category").cat.codes
    df["batter_id"] = df["batter"].astype("category").cat.codes

    return df, features, le_pitch, scaler


def build_sequences(df, features):
    X, Y, P, B = [], [], [], []

    for _, group in df.groupby(["pitcher", "game_pk", "at_bat_number"]):
        feats = group[features].values
        labels = group["pitch_encoded"].values
        p_ids = group["pitcher_id"].values
        b_ids = group["batter_id"].values

        if len(feats) <= SEQUENCE_LEN:
            continue

        for i in range(SEQUENCE_LEN, len(feats)):
            X.append(feats[i-SEQUENCE_LEN:i])
            Y.append(labels[i])
            P.append(p_ids[i-SEQUENCE_LEN:i])
            B.append(b_ids[i-SEQUENCE_LEN:i])

    return np.array(X), np.array(Y), np.array(P), np.array(B)


if __name__ == "__main__":
    df = load_statcast()

    df, features, le_pitch, scaler = preprocess(df)
    X, Y, P, B = build_sequences(df, features)

    # Save artifacts
    np.save("artifacts/processed_X.npy", X)
    np.save("artifacts/processed_Y.npy", Y)
    np.save("artifacts/processed_pitcher.npy", P)
    np.save("artifacts/processed_batter.npy", B)

    pickle.dump(le_pitch, open("artifacts/label_encoder.pkl", "wb"))
    pickle.dump(scaler, open("artifacts/scaler.pkl", "wb"))
    pickle.dump(features, open("artifacts/features.pkl", "wb"))

    print("✓ Preprocessing complete")
    print("Saved sequences to artifacts/")