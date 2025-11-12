import pandas as pd
import numpy as np
import warnings
from pybaseball import statcast, cache
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Masking, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers


SEQ_LEN = 8           # number of previous pitches used to predict next pitch
TEST_SIZE = 0.15
RANDOM_SEED = 42

# -----------------------------------
# Load Data
# -----------------------------------
cache.enable()
warnings.filterwarnings("ignore", category=FutureWarning)

print("Loading Statcast data...")
try:
    df_2024 = pd.read_csv("statcast_2024.csv")
except FileNotFoundError:
    df_2024 = statcast(start_dt='2024-04-01', end_dt='2024-09-30')
    df_2024.to_csv("statcast_2024.csv", index=False)

try:
    df_2023 = pd.read_csv("statcast_2023.csv")
except FileNotFoundError:
    df_2023 = statcast(start_dt='2023-04-01', end_dt='2023-09-30')
    df_2023.to_csv("statcast_2023.csv", index=False)

try:
    df_2022 = pd.read_csv("statcast_2022.csv")
except FileNotFoundError:
    df_2022 = statcast(start_dt='2022-04-01', end_dt='2022-09-30')
    df_2022.to_csv("statcast_2022.csv", index=False)

try:
    df_2021 = pd.read_csv("statcast_2021.csv")
except FileNotFoundError:
    df_2021 = statcast(start_dt='2021-04-01', end_dt='2021-09-30')
    df_2021.to_csv("statcast_2021.csv", index=False)

# Concatenate the three seasons
df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
print(f"Combined data shape: {df.shape}")

# -----------------------------------
# Preprocessing
# -----------------------------------

features = [
    'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'bat_score', 'fld_score'
]
target = ['plate_x', 'plate_z']

df = df.dropna(subset=features + target + ['stand', 'p_throws', 'zone', 'pitch_type'])

# Convert base runners to binary
for base in ['on_1b', 'on_2b', 'on_3b']:
    df[base] = df[base].notna().astype(int)

# Add score difference
df['score_diff'] = df['bat_score'] - df['fld_score']
features.append('score_diff')


# One-hot encode handedness and pitch type
df = pd.get_dummies(df, columns=['stand', 'p_throws'], drop_first=False)
features += [col for col in df.columns if col.startswith('stand_') or col.startswith('p_throws_')]

# Sort by pitcher, game, and pitch order
df = df.sort_values(by=['pitcher', 'game_date', 'at_bat_number', 'pitch_number'])

#Encode pitch type as numeric
le_pitch = LabelEncoder()
df['pitch_encoded'] = le_pitch.fit_transform(df['pitch_type'])
features.append('pitch_encoded')

#Combine all selected features into a single input matrix
X = df[features].fillna(0).astype(float).values

#Targets: plate location coordinates
y = df[['plate_x', 'plate_z']].astype(float).values


# We'll build sequences per pitcher: for each pitcher, sliding windows of SEQ_LEN inputs to predict the next plate_x/z
seqs_X = []
seqs_y = []

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

# scale targets (optional) — here we predict raw plate_x/plate_z, but scaling target can help.
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

# 8) train
history = model.fit(X_train_scaled, y_train_scaled,
                    validation_data=(X_test_scaled, y_test_scaled),
                    epochs=20,
                    batch_size=256)

# 9) evaluate & sample predictions
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae_x = mean_absolute_error(y_test[:,0], y_pred[:,0])
mae_z = mean_absolute_error(y_test[:,1], y_pred[:,1])
rmse_x = np.sqrt(mean_squared_error(y_test[:,0], y_pred[:,0]))
rmse_z = np.sqrt(mean_squared_error(y_test[:,1], y_pred[:,1]))

print(f"MAE plate_x: {mae_x:.3f}, MAE plate_z: {mae_z:.3f}")
print(f"RMSE plate_x: {rmse_x:.3f}, RMSE plate_z: {rmse_z:.3f}")


plt.figure(figsize=(6,6))
plt.scatter(y_test[:,0], y_test[:,1], alpha=0.3, label="True", s=10)
plt.scatter(y_pred[:,0], y_pred[:,1], alpha=0.3, label="Predicted", s=10)
plt.xlabel("plate_x")
plt.ylabel("plate_z")
plt.legend()
plt.title("True vs Predicted Pitch Locations")
plt.show()