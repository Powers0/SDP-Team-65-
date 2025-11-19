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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# -----------------------------------
# Load Data
# -----------------------------------
cache.enable()
warnings.filterwarnings("ignore", category=FutureWarning)

print("Loading Statcast data...")
try:
    df_2024 = pd.read_csv("statcast_2024.csv")
except FileNotFoundError:
    print("Loading 2024 data")
    df_2024 = statcast(start_dt='2024-04-01', end_dt='2024-09-30')
    df_2024.to_csv("statcast_2024.csv", index=False)

try:
    df_2023 = pd.read_csv("statcast_2023.csv")
except FileNotFoundError:
    print("Loading 2023 data")
    df_2023 = statcast(start_dt='2023-04-01', end_dt='2023-09-30')
    df_2023.to_csv("statcast_2023.csv", index=False)

try:
    df_2022 = pd.read_csv("statcast_2022.csv")
except FileNotFoundError:
    print("Loading 2022 data")
    df_2022 = statcast(start_dt='2022-04-01', end_dt='2022-09-30')
    df_2022.to_csv("statcast_2022.csv", index=False)

try:
    df_2021 = pd.read_csv("statcast_2021.csv")
except FileNotFoundError:
    print("Loading 2021 data")
    df_2021 = statcast(start_dt='2021-04-01', end_dt='2021-09-30')
    df_2021.to_csv("statcast_2021.csv", index=False)


# Concatenate the three seasons
df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
#= df_2024.copy()
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

# Convert base runners to binary
for base in ['on_1b', 'on_2b', 'on_3b']:
    df[base] = df[base].notna().astype(int)

dropna_cols = [c for c in features if c not in ['on_1b', 'on_2b', 'on_3b']] + [target, 'stand', 'p_throws', 'zone']
df = df.dropna(subset=dropna_cols)
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
# le_pitch = LabelEncoder()
# df['pitch_encoded'] = le_pitch.fit_transform(df['pitch_type'])
# Define what counts as a "Fastball"
fastball_types = ['FF', 'FC', 'SI']

# Create a new column: 1 if Fastball, 0 if Offspeed
df['binary_target'] = df['pitch_type'].apply(lambda x: 1 if x in fastball_types else 0)

# UPDATE: This is now our label used for the sequences
labels = df['binary_target'].values
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
embedding_dim = 4  # adjustable

# -----------------------------------
# Create sequential data
# -----------------------------------
sequence_length = 5
X_sequences, y_sequences = [], []
pitcher_seq, batter_seq = [], []

count = 0
print(f"Total groups to process: {df.groupby(['pitcher', 'game_pk', 'at_bat_number']).ngroups}")

for _, group in df.groupby(['pitcher', 'game_pk', 'at_bat_number']):
    pitches = group[features].values
    pitchers = group['pitcher_id'].values
    batters = group['batter_id'].values
    labels = group['binary_target'].values
    
    if len(pitches) <= sequence_length:
        continue
    
    count += 1

    for i in range(sequence_length, len(pitches)):
            X_sequences.append(pitches[i-sequence_length:i])
            
            # NEW: Grab the sequence shifted by one step
            # If Input is pitches [0, 1, 2, 3, 4], Target is [1, 2, 3, 4, 5]
            target_sequence = labels[i-sequence_length+1 : i+1]
            y_sequences.append(target_sequence)

            pitcher_seq.append(pitchers[i-sequence_length:i])
            batter_seq.append(batters[i-sequence_length:i])
    
    if count == 0:
        print("ERROR: No sequences were generated. Sequence length might be too long or filtering too strict.")
        exit()

X_sequences = np.array(X_sequences, dtype=np.float32)
y_sequences = np.array(y_sequences, dtype=np.int32)
pitcher_seq = np.array(pitcher_seq, dtype=np.int32)
batter_seq = np.array(batter_seq, dtype=np.int32)
y_cat = to_categorical(y_sequences, num_classes=2)

# -----------------------------------
# Train/test split
# -----------------------------------
split = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:split], X_sequences[split:]
pitcher_train, pitcher_test = pitcher_seq[:split], pitcher_seq[split:]
batter_train, batter_test = batter_seq[:split], batter_seq[split:]
y_train, y_test = y_cat[:split], y_cat[split:]

# -----------------------------------
# Calculate Sample Weights (Smoothed)
# -----------------------------------
print("Calculating smoothed sample weights...")

y_train_integers = np.argmax(y_train, axis=-1)
unique_classes = np.unique(y_train_integers)
from sklearn.utils import class_weight

# Calculate standard balanced weights
raw_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y_train_integers.flatten()
)

# --- THE FIX: SMOOTH THE WEIGHTS ---
# We take the square root to reduce the difference between rare and common classes
# If Fastball was 0.2 and Curve was 5.0 (25x difference),
# Now Fastball is 0.45 and Curve is 2.2 (5x difference). Much safer. 
smoothed_weights = raw_weights ** 0.5 
weight_dict = dict(zip(unique_classes, smoothed_weights))
print(f"Smoothed Class Weights: {weight_dict}")

# Apply the smoothed weights to the sequence matrix
train_sample_weights = np.vectorize(weight_dict.get)(y_train_integers)
# -----------------------------------
# Build LSTM Model with embeddings
# -----------------------------------
print(X_sequences.shape)
input_features = Input(shape=(sequence_length, X_sequences.shape[2]))
input_pitcher = Input(shape=(sequence_length,))
input_batter = Input(shape=(sequence_length,))

pitcher_emb = Embedding(input_dim=num_pitchers, output_dim=embedding_dim)(input_pitcher)
batter_emb = Embedding(input_dim=num_batters, output_dim=embedding_dim)(input_batter)

x = Concatenate(axis=-1)([input_features, pitcher_emb, batter_emb])
x = Masking(mask_value=0.0)(x)
x = LSTM(64, return_sequences=True)(x) 
x = Dropout(0.3)(x)

x = TimeDistributed(Dense(32, activation='relu', ))(x)
x = Dropout(0.3)(x)
output = TimeDistributed(Dense(2, activation='softmax'))(x)

model = Model(inputs=[input_features, input_pitcher, input_batter], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------------
# Train
# -----------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    [X_train, pitcher_train, batter_train],
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop],
    #sample_weight=train_sample_weights,
    verbose=1
)

# -----------------------------------
# Evaluate Binary Model
# -----------------------------------
print("\nEvaluating model...")
loss, acc = model.evaluate([X_test, pitcher_test, batter_test], y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# Predictions
y_pred_probs = model.predict([X_test, pitcher_test, batter_test])

# Flatten results (Sequence -> Individual Pitches)
y_pred = np.argmax(y_pred_probs, axis=-1).flatten()
y_true = np.argmax(y_test, axis=-1).flatten()

# Define Manual Labels (0=Offspeed, 1=Fastball)
# We defined 0 as NOT in fastball_types, and 1 as IN fastball_types
binary_labels = ['Offspeed', 'Fastball']

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=binary_labels, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=binary_labels,
            yticklabels=binary_labels,
            cmap="Blues")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix — Fastball vs. Offspeed")
plt.tight_layout()
plt.show()
# -----------------------------------
# Plot Training History
# -----------------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("LSTM Training vs Validation Accuracy")
plt.show()