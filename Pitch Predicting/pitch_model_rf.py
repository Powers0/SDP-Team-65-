import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pybaseball import statcast, cache
import warnings

#Enable pybaseball caching (speeds up repeated queries)
cache.enable()
warnings.filterwarnings("ignore", category=FutureWarning)

#Load data
print("Loading Statcast data...")
df = statcast(start_dt='2024-04-01', end_dt='2024-05-30')

#Drop rows missing essential features or target
features = [
    'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'bat_score', 'fld_score'
]
target = 'pitch_type'
df = df.dropna(subset=features + [target])

#Convert base runner columns to 0/1 
for base in ['on_1b', 'on_2b', 'on_3b']:
    df[base] = df[base].notna().astype(int)

#Add derived feature: score difference (batter's team minus fielding team)
df['score_diff'] = df['bat_score'] - df['fld_score']

#Keep only the most common MLB pitch types
common_pitches = ['FF', 'SL', 'SI', 'CH', 'CU', 'FC']
df = df[df['pitch_type'].isin(common_pitches)]

#Define final feature set (including derived feature)
features = [
    'balls', 'strikes',
    'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'bat_score', 'fld_score', 'score_diff'
]

#Prepare data
X = df[features].values
y_raw = df[target].values

#Encode target labels (e.g. FF → 0, SL → 1, etc.)
le = LabelEncoder()
y = le.fit_transform(y_raw)

#Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#Random Forest Model
print("Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete.")

#Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix — Pitch Type Prediction (Context Features Only)")
plt.tight_layout()
plt.show()

#Feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), np.array(features)[indices], rotation=45)
plt.tight_layout()
plt.show()