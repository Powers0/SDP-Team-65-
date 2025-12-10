# lstm_statcast_pitch_location.py
import pandas as pd
import numpy as np
from pybaseball import statcast
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from tensorflow.keras import layers, models


SEQ_LEN = 8           # number of previous pitches used to predict next pitch
TEST_SIZE = 0.15
RANDOM_SEED = 42

print("loading data...")
df = statcast(start_dt="2024-03-01", end_dt="2024-11-30")   # pybaseball statcast fetch
print("Downloaded rows:", len(df))

df["z_norm"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])


cols = [
    'game_date', 'pitch_type', 'pitcher', 'batter', 'pitch_number',
    'plate_x', 'plate_z', 'release_speed', 'release_pos_x', 'release_pos_z',
    'pfx_x', 'pfx_z', 'spin_rate', 'events',
    'description', 'game_pk', 'inning', 'balls', 'strikes',
    'stand', 'p_throws', 'zone'
]
df = df[[c for c in cols if c in df.columns]].copy()

# drop pitches missing plate location or pitch_type
df = df.dropna(subset=['plate_x', 'plate_z'])
df = df.reset_index(drop=True)

# encode handedness and pitch_type
df['game_date'] = pd.to_datetime(df['game_date'])
df['pitch_type'] = df['pitch_type'].fillna('UNK')
df['stand'] = df['stand'].fillna('U')
df['p_throws'] = df['p_throws'].fillna('U')

# group by pitcher and at-bats
# sort by pitcher + game_date + pitch_number to preserve order
df = df.sort_values(['pitcher', 'game_date', 'game_pk', 'inning', 'pitch_number']).reset_index(drop=True)

# Select numeric features to use in the LSTM input for each pitch
num_features = ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'spin_rate', 'balls', 'strikes', 'zone']
# ensure these exist, otherwise drop missing columns
num_features = [c for c in num_features if c in df.columns]

# Prepare label encoders / one-hot for categorical features (pitch_type, stand, p_throws)
pitch_le = LabelEncoder()
df['pitch_type_le'] = pitch_le.fit_transform(df['pitch_type'])

# one-hot encoder for batter/pitcher handedness + pitch type
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_to_ohe = ohe.fit_transform(df[['pitch_type_le', 'stand', 'p_throws']])

# concatenate numeric + one-hot into a single feature matrix for each pitch
X_base = df[num_features].fillna(0).astype(float).values
X = np.hstack([X_base, cat_to_ohe])
y = df[['plate_x', 'z_norm']].astype(float).values

# get pitcher group 
grouped = df.groupby('pitcher').indices
for pitcher, idxs in grouped.items():
    idxs = list(idxs)
    # ensure they are in order
    for i in range(len(idxs) - SEQ_LEN):
        window_idx = idxs[i:i+SEQ_LEN]
        target_idx = idxs[i+SEQ_LEN]   # predict the pitch after the window
        seq_feature = X[window_idx]    # shape (SEQ_LEN, features)
        seq_target = y[target_idx]     # shape (2,)
        if np.isnan(seq_feature).any() or np.isnan(seq_target).any():
            continue
        seqs_X.append(seq_feature)
        seqs_y.append(seq_target)

if len(seqs_X) == 0:
    raise RuntimeError("No sequences produced. Try lowering SEQ_LEN or change grouping strategy.")

X_seqs = np.array(seqs_X)  # (n_samples, SEQ_LEN, n_features)
y_seqs = np.array(seqs_y)  # (n_samples, 2)

print("Sequences:", X_seqs.shape, "Targets:", y_seqs.shape)

#  split into train/test by time
X_train, X_test, y_train, y_test = train_test_split(X_seqs, y_seqs, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True)

# scale numeric channels. Because we have sequences and one-hot appended, we scale per-feature across flattened set.
nsamples, seq_len, nfeat = X_train.shape
X_train_flat = X_train.reshape((-1, nfeat))
X_test_flat  = X_test.reshape((-1, nfeat))

scaler = StandardScaler()
scaler.fit(X_train_flat)     # fit only on training
X_train_scaled = scaler.transform(X_train_flat).reshape((-1, seq_len, nfeat))
X_test_scaled  = scaler.transform(X_test_flat).reshape((-1, seq_len, nfeat))

# scale targets — here we predict raw plate_x/plate_z, but scaling target can help.
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled  = y_scaler.transform(y_test)

# build LSTM model
tf.random.set_seed(RANDOM_SEED)
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, nfeat)),
    layers.Masking(mask_value=0.0),           # if padding used later
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(2)   # outputs: plate_x, plate_z
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse',
              metrics=['mae'])

model.summary()


history = model.fit(X_train_scaled, y_train_scaled,
                    validation_data=(X_test_scaled, y_test_scaled),
                    epochs=20,
                    batch_size=256)

# evaluate
eval_res = model.evaluate(X_test_scaled, y_test_scaled, verbose=2)
print("Test loss, test mae (scaled):", eval_res)

# make predictions and invert scale
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# quick metric in original units
mse = np.mean((y_pred - y_test)**2)
mae = np.mean(np.abs(y_pred - y_test))
print(f"Test MSE (plate_x/z): {mse:.4f}, MAE: {mae:.4f}")

# show a few example predictions vs ground truth
for i in range(5):
    print("pred:", y_pred[i], "true:", y_test[i])


