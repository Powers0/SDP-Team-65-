import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pybaseball import statcast
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import warnings
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from pybaseball import cache
cache.enable()

warnings.filterwarnings("ignore", category=FutureWarning)

#1. Load and Prepare Data
# This is the step that is currently failing.
# It would load your CSV into a pandas DataFrame.
df = statcast(start_dt='2024-04-01', end_dt='2024-05-30')

# Define the features and the target variable
features = ['release_speed', 'plate_x', 'plate_z']
target = 'pitch_type'

# Clean the data: remove rows with missing values and keep only the most common pitches
df_model = df.dropna(subset=features + [target])
common_pitches = ['FF', 'SL', 'SI', 'CH', 'CU', 'FC']
df_model = df_model[df_model['pitch_type'].isin(common_pitches)]

# Prepare the data for the model
X = df_model[features].values
y_raw = df_model[target].values

# Convert text labels (like 'FF', 'SL') into numbers
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Scale the features so they have a similar range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#2. Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#3. Train the RandomForest Model
print("Training a RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Training complete.")

#4. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

#5. Visualize the Results
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix for Pitch Type Prediction")
plt.savefig("confusion_matrix_sklearn.png")
plt.show() # Use show() to display the plot directly