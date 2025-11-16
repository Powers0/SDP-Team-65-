import pandas as pd
import numpy as np
import warnings
from pybaseball import statcast, cache
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    Masking, Concatenate, RepeatVector
)
from tensorflow.keras.models import Model


SEQ_LEN = 5
TEST_SIZE = 0.15
RANDOM_SEED = 42

cache.enable()
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------
# Load Statcast Data
# -----------------------------------
def load_statcast(fname, start, end):
    try:
        return pd.read_csv(fname)
    except FileNotFoundError:
        df = statcast(start_dt=start, end_dt=end)
        df.to_csv(fname, index=False)
        return df

df_2024 = load_statcast("statcast_2024.csv", "2024-04-01", "2024-09-30")
df_2023 = load_statcast("statcast_2023.csv", "2023-04-01", "2023-09-30")
df_2022 = load_statcast("statcast_2022.csv", "2022-04-01", "2022-09-30")

df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
print("Data shape:", df.shape)


# -----------------------------------
# Preprocessing
# -----------------------------------
features = [
    "balls", "strikes", "outs_when_up", "inning",
    "on_1b", "on_2b", "on_3b",
    "bat_score", "fld_score"
]
target = ["plate_x", "plate_z"]

df = df.dropna(subset=features + target +
               ["stand", "p_throws", "zone", "pitch_type", "pitcher", "batter"])

# Base runner binary
for base in ["on_1b", "on_2b", "on_3b"]:
    df[base] = df[base].notna().astype(int)

# Score diff
df["score_diff"] = df["bat_score"] - df["fld_score"]
features.append("score_diff")

# Categorical one-hot
df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)
features += [c for c in df.columns if c.startswith("stand_") or c.startswith("p_throws_")]

# Pitch type encoding
le_pitch = LabelEncoder()
df["pitch_encoded"] = le_pitch.fit_transform(df["pitch_type"])
features.append("pitch_encoded")

# Pitcher / Batter IDs for embeddings
df["pitcher_id"] = df["pitcher"].astype("category").cat.codes
df["batter_id"] = df["batter"].astype("category").cat.codes

num_pitchers = df["pitcher_id"].nunique()
num_batters  = df["batter_id"].nunique()

# Sorting
df = df.sort_values(by=["pitcher", "game_date", "at_bat_number", "pitch_number"])

# Feature matrix
X = df[features].fillna(0).astype(float).values
y = df[["plate_x", "plate_z"]].astype(float).values


# -----------------------------------
# Build Sequences
# -----------------------------------
seqs_X, seqs_y = [], []
seqs_pitcher, seqs_batter = [], []

grouped = df.groupby("pitcher").indices

for pitcher, idxs in grouped.items():
    idxs = list(idxs)
    for i in range(len(idxs) - SEQ_LEN):
        win = idxs[i:i+SEQ_LEN]
        tgt = idxs[i+SEQ_LEN]

        seqs_X.append(X[win])
        seqs_y.append(y[tgt])

        seqs_pitcher.append(df.iloc[tgt]["pitcher_id"])
        seqs_batter.append(df.iloc[tgt]["batter_id"])

X_seqs = np.array(seqs_X)
y_seqs = np.array(seqs_y)
pitcher_ids = np.array(seqs_pitcher)
batter_ids = np.array(seqs_batter)

print("Sequences:", X_seqs.shape, "Targets:", y_seqs.shape)


# -----------------------------------
# Train/Test Split
# -----------------------------------
(
    X_train, X_test,
    y_train, y_test,
    p_train, p_test,
    b_train, b_test
) = train_test_split(
    X_seqs, y_seqs,
    pitcher_ids, batter_ids,
    test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
)

# Flatten and scale features
nsamples, seq_len, nfeat = X_train.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(
    X_train.reshape((-1, nfeat))
).reshape((-1, seq_len, nfeat))

X_test_scaled = scaler.transform(
    X_test.reshape((-1, nfeat))
).reshape((-1, seq_len, nfeat))

# Scale targets
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# -----------------------------------
# Model with Embeddings
# -----------------------------------
tf.random.set_seed(RANDOM_SEED)

seq_input = Input(shape=(SEQ_LEN, nfeat))

pitcher_input = Input(shape=(), dtype="int32")
batter_input  = Input(shape=(), dtype="int32")

pitcher_emb_dim = 32
batter_emb_dim  = 16

pitcher_emb = Embedding(num_pitchers, pitcher_emb_dim)(pitcher_input)
batter_emb  = Embedding(num_batters,  batter_emb_dim)(batter_input)

pitcher_emb_rep = RepeatVector(SEQ_LEN)(pitcher_emb)
batter_emb_rep  = RepeatVector(SEQ_LEN)(batter_emb)

x = Concatenate(axis=-1)([seq_input, pitcher_emb_rep, batter_emb_rep])

x = LSTM(128, return_sequences=True, recurrent_dropout=0.1)(x)
x = Dropout(0.2)(x)
x = LSTM(64)(x)
x = Dense(32, activation="relu")(x)
output = Dense(2)(x)

model = Model(
    inputs=[seq_input, pitcher_input, batter_input],
    outputs=output
)

model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss="mse",
              metrics=["mae"])

print(model.summary())


# -----------------------------------
# Train
# -----------------------------------
history = model.fit(
    [X_train_scaled, p_train, b_train],
    y_train_scaled,
    validation_data=([X_test_scaled, p_test, b_test], y_test_scaled),
    epochs=30,
    batch_size=256
)


# -----------------------------------
# Evaluate
# -----------------------------------
eval_res = model.evaluate(
    [X_test_scaled, p_test, b_test], y_test_scaled, verbose=2
)

print("Test loss, test MAE (scaled):", eval_res)

y_pred_scaled = model.predict([X_test_scaled, p_test, b_test])
y_pred = y_scaler.inverse_transform(y_pred_scaled)

mse = np.mean((y_pred - y_test)**2)
mae = np.mean(np.abs(y_pred - y_test))

print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

# Per-axis
mae_x = mean_absolute_error(y_test[:,0], y_pred[:,0])
mae_z = mean_absolute_error(y_test[:,1], y_pred[:,1])
rmse_x = np.sqrt(mean_squared_error(y_test[:,0], y_pred[:,0]))
rmse_z = np.sqrt(mean_squared_error(y_test[:,1], y_pred[:,1]))

print(f"MAE plate_x: {mae_x:.3f}, MAE plate_z: {mae_z:.3f}")
print(f"RMSE plate_x: {rmse_x:.3f}, RMSE plate_z: {rmse_z:.3f}")

for i in range(5):
    print("pred:", y_pred[i], "true:", y_test[i])


plt.figure(figsize=(6,6))
plt.scatter(y_test[:,0], y_test[:,1], alpha=0.3, s=10, label="True")
plt.scatter(y_pred[:,0], y_pred[:,1], alpha=0.3, s=10, label="Pred")
plt.xlabel("plate_x")
plt.ylabel("plate_z")
plt.legend()
plt.show()