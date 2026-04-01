import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


TEST_SIZE = 0.15
RANDOM_SEED = 42

CSV_DIR = "../../csv data"
ARTIFACTS = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"

YEARS = [2021, 2022, 2023, 2024]

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

MISS_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
FOUL_DESCRIPTIONS = {"foul", "foul_tip"}
FAIR_DESCRIPTIONS = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}


ALL_PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "KC", "ST", "SV", "CS", "FO", "KN", "EP"]

CLASSES = ["miss", "foul", "fair"]  # index 0, 1, 2

def load_statcast():
    dfs = []
    for y in YEARS:
        path = f"{CSV_DIR}/statcast_full_{y}.csv"
        print("Loading:", path)
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    needed = [
        "balls", "strikes", "outs_when_up", "inning",
        "bat_score", "fld_score",
        "plate_x", "plate_z", "stand", "p_throws",
        "pitch_type", "description"
    ]

    df = df.dropna(subset=needed)
    df = df[df["pitch_type"].isin(ALL_PITCH_TYPES)].copy()
    df = df[df["description"].isin(SWING_DESCRIPTIONS)].copy()

    for base in ["on_1b", "on_2b", "on_3b"]:
        df[base] = df[base].notna().astype(int)

    df["score_diff"] = df["bat_score"] - df["fld_score"]
    df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)


    def label(desc):
        if desc in MISS_DESCRIPTIONS:
            return 0
        elif desc in FOUL_DESCRIPTIONS:
            return 1
        else:
            return 2

    df["contact_outcome"] = df["description"].apply(label)

    return df

def build_pitch_type_onehot(df):
    pt = pd.Categorical(df["pitch_type"], categories=ALL_PITCH_TYPES)
    return pd.get_dummies(pt).values.astype(float)


def build_location_features(df):
    x = df["plate_x"].values
    z = df["plate_z"].values
    dist_to_center = np.sqrt((x - 0.0) ** 2 + (z - 2.5) ** 2)
    is_strike = ((np.abs(x) <= 0.83) & (z >= 1.5) & (z <= 3.5)).astype(float)
    return np.column_stack([x, z, dist_to_center, is_strike])


def build_context_features(df):
    stand_cols   = [c for c in df.columns if c.startswith("stand_")]
    pthrows_cols = [c for c in df.columns if c.startswith("p_throws_")]
    cols = ["balls", "strikes", "outs_when_up", "inning", "score_diff",
            "on_1b", "on_2b", "on_3b"] + stand_cols + pthrows_cols
    return df[cols].values.astype(float), cols



def build_player_ids(df):
    pitcher_le = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le  = pickle.load(open(SHARED_DIR + "batter_le.pkl", "rb"))

    p_ids = df["pitcher"].map(lambda x: pitcher_le.transform([x])[0] if x in pitcher_le.classes_ else 0).values
    b_ids = df["batter"].map(lambda x: batter_le.transform([x])[0] if x in batter_le.classes_ else 0).values
    return p_ids.astype(np.int32), b_ids.astype(np.int32), pitcher_le, batter_le



if __name__ == "__main__":
    print("Loading statcast data...")
    df = load_statcast()

    print("Preprocessing...")
    df = preprocess(df)

    print("Building features...")
    PT = build_pitch_type_onehot(df)
    LOC = build_location_features(df)
    CTX, ctx_feature_names = build_context_features(df)
    P, B, pitcher_le, batter_le = build_player_ids(df)

    y = df["contact_outcome"].values.astype(int)

    print(f"Dataset size: {len(y)} swings")
    print(f"Miss: {(y==0).mean():.3f}  Foul: {(y==1).mean():.3f}  Fair: {(y==2).mean():.3f}")

    print("Train/test split...")
    PT_train, PT_test, LOC_train, LOC_test, CTX_train, CTX_test, P_train, P_test, B_train, B_test, y_train, y_test = train_test_split(
        PT, LOC, CTX, P, B, y,
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
    np.save(ARTIFACTS + "P_train.npy", P_train)
    np.save(ARTIFACTS + "P_test.npy",  P_test)
    np.save(ARTIFACTS + "B_train.npy", B_train)
    np.save(ARTIFACTS + "B_test.npy",  B_test)

    pickle.dump(ctx_feature_names, open(ARTIFACTS + "ctx_features.pkl", "wb"))


    pickle.dump(ALL_PITCH_TYPES, open(ARTIFACTS + "pitch_types.pkl", "wb"))
    pickle.dump(CLASSES, open(ARTIFACTS + "classes.pkl", "wb"))

    print("\n✓ Contact outcome dataset built.")
    print("PT_train:", PT_train.shape, "LOC_train:", LOC_train.shape, "y_train:", y_train.shape)
