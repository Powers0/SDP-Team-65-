# pitch_type_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from pybaseball import statcast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================
# 1. Load Data
# ==========================
# Pull one month of Statcast data
df = statcast(start_dt='2024-04-01', end_dt='2024-04-30')

# Filter for relevant columns and drop missing values
df = df.dropna(subset=['pitch_type', 'release_speed', 'release_spin_rate', 'p_throws', 'stand', 'balls', 'strikes', 'outs_when_up', 'inning'])

features = ['balls', 'strikes', 'outs_when_up', 'inning', 'p_throws', 'stand', 'release_speed', 'release_spin_rate']
X = df[features]
y = df['pitch_type']

# It is necessary to LabelEncode p_throw, Stand, and pitch_Type to take these values from categorical to numerical
for col in ['p_throws', 'stand']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode pitch types (target)
y_le = LabelEncoder()
y_encoded = y_le.fit_transform(y)

#Standardizes each feature by removing its mean and dividing by its standard deviation, 
# making each continuous feature have a mean of around 0 and a standard deviation of 1
#-This is important because gradient-based optimizers are able to converge faster when inputs are on similar scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)  # long for classification
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ==========================
# 2. Define Model
# ==========================
#2 Layer MLP (Multi Layer Perception) model, good for relationships between numeric features
#ReLu introduces non-linearity
#nnLinear(input_dim, 64) maps input vector -> vector length 64
#Then, Linear 64 -> 32, Then 32 -> num_classes outputs
class PitchTypeModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PitchTypeModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

model = PitchTypeModel(X_train_t.shape[1], len(y_le.classes_))

# ==========================
# 3. Training
# ==========================
#Best for multi-class classification where targets are class indices
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad() #pytorch accumulates gradients by default, so we must zero them each training iteration
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ==========================
# 4. Evaluation
# ==========================
#Going to implement a different form of accuracy 
model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == y_test_t).float().mean()
    print(f"Test Accuracy: {accuracy.item()*100:.2f}%")