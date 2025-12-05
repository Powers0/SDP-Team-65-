import pandas as pd
import numpy as np
import warnings
from pybaseball import statcast, cache
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Masking, Concatenate, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -----------------------------------------
# GLOBAL TOGGLES
# -----------------------------------------
USE_CLASS_WEIGHTS = True   # <--- Toggle ON/OFF
SEQUENCE_LENGTH   = 5       # You may later increase to 10

# -----------------------------------
# Load Statcast Data
# -----------------------------------
cache.enable()
warnings.filterwarnings("ignore", category=FutureWarning)

print("Loading Statcast data...")
try:
    df_2024 = pd.read_csv("statcast_2024.csv")
except FileNotFoundError:
    print("Downloading 2024")
    df_2024 = statcast(start_dt='2024-04-01', end_dt='2024-09-30')
    df_2024.to_csv("statcast_2024.csv", index=False)

try:
    df_2023 = pd.read_csv("statcast_2023.csv")
except FileNotFoundError:
    print("Downloading 2023")
    df_2023 = statcast(start_dt='2023-04-01', end_dt='2023-09-30')
    df_2023.to_csv("statcast_2023.csv", index=False)

try:
    df_2022 = pd.read_csv("statcast_2022.csv")
except FileNotFoundError:
    print("Downloading 2022")
    df_2022 = statcast(start_dt='2022-04-01', end_dt='2022-09-30')
    df_2022.to_csv("statcast_2022.csv", index=False)

try:
    df_2021 = pd.read_csv("statcast_2021.csv")
except FileNotFoundError:
    print("Downloading 2021")
    df_2021 = statcast(start_dt='2021-04-01', end_dt='2021-09-30')
    df_2021.to_csv("statcast_2021.csv", index=False)

# Use last 3 seasons
df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
print(f"Combined data shape: {df.shape}")

# -----------------------------------
# Base Features
# -----------------------------------
features = [
    'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'bat_score', 'fld_score'
]

target = 'pitch_type'

# Convert runners to binary
for base in ['on_1b', 'on_2b', 'on_3b']:
    df[base] = df[base].notna().astype(int)

# Clean NA rows
dropna_cols = [c for c in features if c not in ['on_1b', 'on_2b', 'on_3b']] + ['stand', 'p_throws', 'zone', target]
df = df.dropna(subset=dropna_cols)

# Score difference feature
df['score_diff'] = df['bat_score'] - df['fld_score']
features.append('score_diff')

# Keep common pitch types
common_pitches = ['FF', 'SL', 'SI', 'CH', 'CU', 'FC', 'ST']
df = df[df['pitch_type'].isin(common_pitches)]

# One-hot handedness
df = pd.get_dummies(df, columns=['stand', 'p_throws'], drop_first=False)
features += [c for c in df.columns if c.startswith('stand_') or c.startswith('p_throws_')]

# Sort data
df = df.sort_values(by=['pitcher', 'game_date', 'at_bat_number', 'pitch_number'])

# Label encode target
le_pitch = LabelEncoder()
df['pitch_encoded'] = le_pitch.fit_transform(df['pitch_type'])

# -----------------------------------
# Add Previous Pitch Type + Zone
# -----------------------------------
df['previous_pitch'] = df.groupby(['pitcher', 'game_pk', 'at_bat_number'])['pitch_type'].shift(1)
df['previous_zone']  = df.groupby(['pitcher', 'game_pk', 'at_bat_number'])['zone'].shift(1)

df['previous_pitch'] = df['previous_pitch'].fillna('None')
df['previous_zone']  = df['previous_zone'].fillna(-1)

df = pd.get_dummies(df, columns=['previous_pitch'], drop_first=False)
features += [c for c in df.columns if c.startswith('previous_pitch_')]
features.append('previous_zone')

# -----------------------------------------
# NEW: Add PREVIOUS RELEASE METRICS
# -----------------------------------------
release_cols = ['release_speed', 'release_pos_x', 'release_pos_z']

for col in release_cols:
    df['prev_' + col] = df.groupby(['pitcher', 'game_pk', 'at_bat_number'])[col].shift(1)

df[['prev_release_speed', 'prev_release_pos_x', 'prev_release_pos_z']] = \
    df[['prev_release_speed', 'prev_release_pos_x', 'prev_release_pos_z']].fillna(0.0)

features += ['prev_release_speed', 'prev_release_pos_x', 'prev_release_pos_z']

# -----------------------------------------
# NEW: Add CURRENT RELEASE METRICS
# -----------------------------------------
current_release_features = [
    'release_speed',
    'release_pos_x',
    'release_pos_z'
]

if 'release_spin_rate' in df.columns:
    current_release_features.append('release_spin_rate')

features += current_release_features

# -----------------------------------
# Scale Numeric Features
# -----------------------------------
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# -----------------------------------
# Pitcher/Batter Embeddings
# -----------------------------------
df['pitcher_id'] = df['pitcher'].astype('category').cat.codes
df['batter_id']  = df['batter'].astype('category').cat.codes

num_pitchers = df['pitcher_id'].nunique()
num_batters  = df['batter_id'].nunique()
embedding_dim = 4

# -----------------------------------
# Create Sequential Data
# -----------------------------------
X_sequences, y_sequences = [], []
pitcher_seq, batter_seq = [], []

for _, group in df.groupby(['pitcher', 'game_pk', 'at_bat_number']):
    feats = group[features].values
    pitch_ids = group['pitcher_id'].values
    bat_ids   = group['batter_id'].values
    labels    = group['pitch_encoded'].values

    if len(feats) <= SEQUENCE_LENGTH:
        continue

    for i in range(SEQUENCE_LENGTH, len(feats)):
        X_sequences.append(feats[i-SEQUENCE_LENGTH:i])

        # Sequence-shifted labels
        target_seq = labels[i-SEQUENCE_LENGTH+1 : i+1]
        y_sequences.append(target_seq)

        pitcher_seq.append(pitch_ids[i-SEQUENCE_LENGTH:i])
        batter_seq.append(bat_ids[i-SEQUENCE_LENGTH:i])

X_sequences = np.array(X_sequences, dtype=np.float32)
y_sequences = np.array(y_sequences, dtype=np.int32)
pitcher_seq = np.array(pitcher_seq, dtype=np.int32)
batter_seq  = np.array(batter_seq, dtype=np.int32)

y_cat = to_categorical(y_sequences, num_classes=len(le_pitch.classes_))

# -----------------------------------
# Train/Test Split
# -----------------------------------
split = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:split], X_sequences[split:]
pitcher_train, pitcher_test = pitcher_seq[:split], pitcher_seq[split:]
batter_train, batter_test   = batter_seq[:split], batter_seq[split:]
y_train, y_test             = y_cat[:split], y_cat[split:]

# -----------------------------------
# CLASS WEIGHT TOGGLE
# -----------------------------------
if USE_CLASS_WEIGHTS:
    print("Using class weights...")
    y_int = np.argmax(y_train, axis=-1)
    classes = np.unique(y_int)

    raw_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_int.flatten()
    )

    smoothed = raw_weights ** 0.5
    weight_dict = dict(zip(classes, smoothed))

    train_sample_weights = np.vectorize(weight_dict.get)(y_int)
else:
    print("Training WITHOUT class weights")
    train_sample_weights = None

# -----------------------------------
# Build Model
# -----------------------------------
input_features = Input(shape=(SEQUENCE_LENGTH, X_train.shape[2]))
input_pitcher  = Input(shape=(SEQUENCE_LENGTH,))
input_batter   = Input(shape=(SEQUENCE_LENGTH,))

pitcher_emb = Embedding(num_pitchers, embedding_dim)(input_pitcher)
batter_emb  = Embedding(num_batters,  embedding_dim)(input_batter)

x = Concatenate(axis=-1)([input_features, pitcher_emb, batter_emb])
x = Masking(mask_value=0.0)(x)

x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.5)(x)

x = TimeDistributed(Dense(64, activation='relu'))(x)
x = Dropout(0.5)(x)

output = TimeDistributed(Dense(len(le_pitch.classes_), activation='softmax'))(x)

model = Model([input_features, input_pitcher, input_batter], output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------------
# Train Model
# -----------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    [X_train, pitcher_train, batter_train],
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    sample_weight=train_sample_weights,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------------
# Evaluation
# -----------------------------------
loss, acc = model.evaluate([X_test, pitcher_test, batter_test], y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

y_pred_probs = model.predict([X_test, pitcher_test, batter_test])
y_pred = np.argmax(y_pred_probs, axis=-1).flatten()
y_true = np.argmax(y_test, axis=-1).flatten()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le_pitch.classes_, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le_pitch.classes_,
            yticklabels=le_pitch.classes_,
            cmap="Blues")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix — LSTM Pitch Type Prediction")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("LSTM Training vs Validation Accuracy")
plt.show()
