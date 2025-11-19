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
target = 'pitch_type'

df = df.dropna(subset=features + [target, 'stand', 'p_throws', 'zone'])

# Convert base runners to binary
for base in ['on_1b', 'on_2b', 'on_3b']:
    df[base] = df[base].notna().astype(int)

# Add score difference
df['score_diff'] = df['bat_score'] - df['fld_score']
features.append('score_diff')

# Limit to common pitch types
common_pitches = ['FF', 'SL', 'SI', 'CH', 'CU', 'FC', 'ST']
df = df[df['pitch_type'].isin(common_pitches)]

# One-hot encode handedness
df = pd.get_dummies(df, columns=['stand', 'p_throws'], drop_first=False)
features += [col for col in df.columns if col.startswith('stand_') or col.startswith('p_throws_')]

# Sort by pitcher, game, and pitch order
df = df.sort_values(by=['pitcher', 'game_date', 'at_bat_number', 'pitch_number'])

# Encode target
le_pitch = LabelEncoder()
df['pitch_encoded'] = le_pitch.fit_transform(df['pitch_type'])

# -----------------------------------
# Add previous pitch and previous zone
# -----------------------------------
df['previous_pitch'] = df.groupby(['pitcher', 'game_pk', 'at_bat_number'])['pitch_type'].shift(1)
df['previous_zone'] = df.groupby(['pitcher', 'game_pk', 'at_bat_number'])['zone'].shift(1)
df['previous_pitch'] = df['previous_pitch'].fillna('None')
df['previous_zone'] = df['previous_zone'].fillna(-1)

# One-hot encode previous pitch
df = pd.get_dummies(df, columns=['previous_pitch'], drop_first=False)
features += [col for col in df.columns if col.startswith('previous_pitch_')]
features.append('previous_zone')

# Standardize numeric features
scaler = StandardScaler()
numeric_features = [f for f in features if f != 'previous_zone'] + ['previous_zone']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# -----------------------------------
# Encode pitcher and batter IDs for embeddings
# -----------------------------------
df['pitcher_id'] = df['pitcher'].astype('category').cat.codes
df['batter_id'] = df['batter'].astype('category').cat.codes
num_pitchers = df['pitcher_id'].nunique()
num_batters = df['batter_id'].nunique()
embedding_dim = 8  # adjustable

# -----------------------------------
# Create sequential data
# -----------------------------------
sequence_length = 5
X_sequences, y_sequences = [], []
pitcher_seq, batter_seq = [], []

for _, group in df.groupby(['pitcher', 'game_pk', 'at_bat_number']):
    pitches = group[features].values
    pitchers = group['pitcher_id'].values
    batters = group['batter_id'].values
    labels = group['pitch_encoded'].values
    
    if len(pitches) <= sequence_length:
        continue
    
    for i in range(sequence_length, len(pitches)):
        X_sequences.append(pitches[i-sequence_length:i])
        y_sequences.append(labels[i])
        pitcher_seq.append(pitchers[i-sequence_length:i])
        batter_seq.append(batters[i-sequence_length:i])

X_sequences = np.array(X_sequences, dtype=np.float32)
y_sequences = np.array(y_sequences, dtype=np.int32)
pitcher_seq = np.array(pitcher_seq, dtype=np.int32)
batter_seq = np.array(batter_seq, dtype=np.int32)
y_cat = to_categorical(y_sequences, num_classes=len(le_pitch.classes_))

# -----------------------------------
# Train/test split
# -----------------------------------
split = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:split], X_sequences[split:]
pitcher_train, pitcher_test = pitcher_seq[:split], pitcher_seq[split:]
batter_train, batter_test = batter_seq[:split], batter_seq[split:]
y_train, y_test = y_cat[:split], y_cat[split:]

# -----------------------------------
# Build LSTM Model with embeddings
# -----------------------------------
input_features = Input(shape=(sequence_length, X_sequences.shape[2]))
input_pitcher = Input(shape=(sequence_length,))
input_batter = Input(shape=(sequence_length,))

pitcher_emb = Embedding(input_dim=num_pitchers, output_dim=embedding_dim)(input_pitcher)
batter_emb = Embedding(input_dim=num_batters, output_dim=embedding_dim)(input_batter)

x = Concatenate(axis=-1)([input_features, pitcher_emb, batter_emb])
x = Masking(mask_value=0.0)(x)
x = LSTM(128, return_sequences=False)(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(len(le_pitch.classes_), activation='softmax')(x)

model = Model(inputs=[input_features, input_pitcher, input_batter], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------------
# Train
# -----------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    [X_train, pitcher_train, batter_train],
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------------
# Evaluate
# -----------------------------------
print("\nEvaluating model...")
loss, acc = model.evaluate([X_test, pitcher_test, batter_test], y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# -----------------------------------
# Save predicted pitch type probabilities
# -----------------------------------
print("\nSaving pitch type probabilities for use in pitch location model...")
y_pred_probs_full = model.predict([X_sequences, pitcher_seq, batter_seq], batch_size=256, verbose=1)
np.save("pitch_type_probs.npy", y_pred_probs_full)
print(f"Saved pitch type probabilities shape: {y_pred_probs_full.shape}")

# -----------------------------------
# Predictions & Evaluation
# -----------------------------------
y_pred = np.argmax(y_pred_probs_full[split:], axis=1)
y_true = np.argmax(y_cat[split:], axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le_pitch.classes_, zero_division=0))

# Confusion Matrix
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

# Training history
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("LSTM Training vs Validation Accuracy")
plt.show()