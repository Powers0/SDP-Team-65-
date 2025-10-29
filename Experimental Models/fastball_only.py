# fastball_only_test.py

import torch
import torch.nn as nn
import torch.optim as optim
from pybaseball import statcast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================
# 1. Load Statcast data
# ==========================
df = statcast(start_dt='2024-04-01', end_dt='2024-05-30')

df = df.dropna(subset=['pitch_type', 'balls', 'strikes', 'outs_when_up', 'inning', 'bat_score', 'fld_score'])
remove_pitches = ['CS', 'EP', 'FA', 'FO', 'KN', 'PO', 'SC']
df = df[~df['pitch_type'].isin(remove_pitches)]

features = ['balls', 'strikes', 'outs_when_up', 'inning', 'on_1b', 'on_2b', 'on_3b', 'bat_score', 'fld_score']
X = df[features].copy()
y = df['pitch_type'].values

X['on_1b'] = X['on_1b'].notna().astype(int)
X['on_2b'] = X['on_2b'].notna().astype(int)
X['on_3b'] = X['on_3b'].notna().astype(int)

# Encode target
y_le = LabelEncoder()
y_encoded = y_le.fit_transform(y)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# ==========================
# 2. Extreme fastball oversampling
# ==========================
# Find the encoded label for 'FF' (four-seam fastball)
try:
    ff_label = np.where(y_le.classes_ == 'FF')[0][0]
except IndexError:
    raise ValueError("No 'FF' (four-seam fastball) pitch type found in dataset.")

# Keep only fastballs in the training set
mask_ff = (y_train == ff_label)
X_train_ff = X_train[mask_ff]
y_train_ff = y_train[mask_ff]

print(f"Training only on fastballs: {len(X_train_ff)} samples (out of {len(X_train)})")

# ==========================
# 3. Model Definition
# ==========================
class PitchTypeModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = PitchTypeModel(input_dim=X_train.shape[1], num_classes=len(y_le.classes_))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
X_train_t = torch.tensor(X_train_ff, dtype=torch.float32)
y_train_t = torch.tensor(y_train_ff, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ==========================
# 4. Train on ONLY fastballs
# ==========================
epochs = 30
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ==========================
# 5. Evaluate on real test data
# ==========================
model.eval()
with torch.no_grad():
    preds = model(X_test_t)
    pred_classes = torch.argmax(preds, dim=1)
    acc = (pred_classes == y_test_t).float().mean().item()
    print(f"\nTest Accuracy (should be ~majority baseline): {acc*100:.2f}%")

# Check which class dominates predictions
pred_counts = pd.Series(pred_classes.numpy()).value_counts()
print("\nPredicted class distribution (should be all FF):")
for label_idx, count in pred_counts.items():
    print(f"{y_le.classes_[label_idx]}: {count}")

# Confusion matrix and report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_t, pred_classes))
print("\nClassification Report:")
print(classification_report(y_test_t, pred_classes, target_names=list(y_le.classes_)))