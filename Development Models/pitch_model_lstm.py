import pandas as pd
import numpy as np
import warnings
from pybaseball import statcast, cache
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------
# Load Data
# -----------------------------------
cache.enable()
warnings.filterwarnings("ignore", category=FutureWarning)

print("Loading Statcast data...")
df = statcast(start_dt='2024-04-01', end_dt='2024-05-30')

# -----------------------------------
# Preprocessing
# -----------------------------------
features = [
    'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'bat_score', 'fld_score'
]
target = 'pitch_type'

df = df.dropna(subset=features + [target, 'stand', 'p_throws'])

# Convert base runners to binary
for base in ['on_1b', 'on_2b', 'on_3b']:
    df[base] = df[base].notna().astype(int)

# Add score difference
df['score_diff'] = df['bat_score'] - df['fld_score']
features.append('score_diff')

# Limit to common pitch types
common_pitches = ['FF', 'SL', 'SI', 'CH', 'CU', 'FC', 'ST', 'FS']
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
# Create sequential data for LSTM
# -----------------------------------
sequence_length = 5  # how many previous pitches to use
X_sequences = []
y_sequences = []

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Group by pitcher & at-bat to form sequences
for _, group in df.groupby(['pitcher', 'game_pk', 'at_bat_number']):
    pitches = group[features].values
    labels = group['pitch_encoded'].values
    if len(pitches) <= sequence_length:
        continue
    for i in range(sequence_length, len(pitches)):
        X_sequences.append(pitches[i-sequence_length:i])
        y_sequences.append(labels[i])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# One-hot encode the target
y_cat = to_categorical(y_sequences, num_classes=len(le_pitch.classes_))

# Train/test split
split = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:split], X_sequences[split:]
y_train, y_test = y_cat[:split], y_cat[split:]

# -----------------------------------
# Build LSTM Model
# -----------------------------------
model = Sequential([
    Masking(mask_value=0.0, input_shape=(sequence_length, X_train.shape[2])),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le_pitch.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------------
# Train the Model
# -----------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nTraining LSTM model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------------
# Evaluate
# -----------------------------------
print("\nEvaluating model...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

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

# -----------------------------------
# Plot Training History
# -----------------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("LSTM Training vs Validation Accuracy")
plt.show()