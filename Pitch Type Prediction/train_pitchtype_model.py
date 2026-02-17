

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from pitchtype_architecture import build_pitch_model
from pathlib import Path


 # Make artifacts path relative to this script (stable across machines)
HERE = Path(__file__).resolve().parent
ARTIFACTS_DIR = HERE / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Load dataset artifacts
# ---------------------------------------------------
print("Loading processed datasets...")

X = np.load(ARTIFACTS_DIR / "processed_X.npy")
Y = np.load(ARTIFACTS_DIR / "processed_Y.npy")
P = np.load(ARTIFACTS_DIR / "processed_pitcher.npy")
B = np.load(ARTIFACTS_DIR / "processed_batter.npy")

le_pitch = pickle.load(open(ARTIFACTS_DIR / "label_encoder.pkl", "rb"))
features_list = pickle.load(open(ARTIFACTS_DIR / "features.pkl", "rb"))

num_features = len(features_list)
num_classes = len(le_pitch.classes_)

print("Dataset Loaded:")
print("X:", X.shape)
print("Y:", Y.shape)
print("P:", P.shape)
print("B:", B.shape)
print("Features:", num_features)

# ---------------------------------------------------
# Train/Test split
# ---------------------------------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
P_train, P_test = P[:split], P[split:]
B_train, B_test = B[:split], B[split:]
Y_train, Y_test = Y[:split], Y[split:]

Y_train_cat = to_categorical(Y_train, num_classes=num_classes)
Y_test_cat = to_categorical(Y_test, num_classes=num_classes)

# ---------------------------------------------------
# Build Model
# ---------------------------------------------------
num_pitchers = P.max() + 1
num_batters = B.max() + 1

model = build_pitch_model(
    num_features=num_features,
    num_pitchers=num_pitchers,
    num_batters=num_batters,
    embed_dim=8,
    num_classes=num_classes
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------------------------------
# Train
# ---------------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    [X_train, P_train, B_train],
    Y_train_cat,
    validation_split=0.2,
    epochs=12,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------------------
# Evaluate
# ---------------------------------------------------
print("\nEvaluating model...")
loss, acc = model.evaluate([X_test, P_test, B_test], Y_test_cat, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# Classification report & confusion matrix
Y_pred_probs = model.predict([X_test, P_test, B_test])
Y_pred = np.argmax(Y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, target_names=le_pitch.classes_, zero_division=0))

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le_pitch.classes_,
            yticklabels=le_pitch.classes_,
            cmap="Blues")
plt.title("Confusion Matrix — Pitch Type Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Save pitch type probabilities for pitch-location model
# ---------------------------------------------------
print("\nSaving pitch type probabilities for location model...")

Y_pred_probs_full = model.predict([X, P, B], batch_size=256)
np.save(ARTIFACTS_DIR / "pitch_type_probs.npy", Y_pred_probs_full)

print("Saved:", ARTIFACTS_DIR / "pitch_type_probs.npy")

# ---------------------------------------------------
# Save trained model
# ---------------------------------------------------
model.save(ARTIFACTS_DIR / "pitchtype_model.keras")
print("Saved model:", ARTIFACTS_DIR / "pitchtype_model.keras")