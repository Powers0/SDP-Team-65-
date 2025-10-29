# pitch_type_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from pybaseball import statcast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import warnings
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore", category=FutureWarning)



# ==========================
# 1. Load Data
# ==========================
# Pull one month of Statcast data
df = statcast(start_dt='2024-04-01', end_dt='2024-05-30')

# Filter for relevant columns and drop missing values
#p_throws, #stand, #batter, #pitcher
df = df.dropna(subset=['pitch_type', 'balls', 'strikes', 'outs_when_up', 'inning', 'bat_score', 'fld_score'])

remove_pitches = ['CS', 'EP', 'FA', 'FO', 'KN', 'PO', 'SC']
df = df[~df['pitch_type'].isin(remove_pitches)]

features = ['balls', 'strikes', 'outs_when_up', 'inning', 'on_1b', 'on_2b', 'on_3b', 'bat_score', 'fld_score']
X = df[features].copy()
y = df['pitch_type'].values

# Convert base runners from IDs → binary flags
X['on_1b'] = X['on_1b'].notna().astype(int)
X['on_2b'] = X['on_2b'].notna().astype(int)
X['on_3b'] = X['on_3b'].notna().astype(int)


# Encode pitch types (target)
y_le = LabelEncoder()
y_encoded = y_le.fit_transform(y)


#Standardizes each feature by removing its mean and dividing by its standard deviation, 
# making each continuous feature have a mean of around 0 and a standard deviation of 1
#-This is important because gradient-based optimizers are able to converge faster when inputs are on similar scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)


#Compute weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
print(dict(zip(y_le.classes_, class_weights.tolist())))

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
# ==========================
# 2. Define Model
# ==========================
#2 Layer MLP (Multi Layer Perception) model, good for relationships between numeric features
#ReLu introduces non-linearity
#nnLinear(input_dim, 64) maps input vector -> vector length 64
#Then, Linear 64 -> 32, Then 32 -> num_classes outputs


#memorize different batter/pitcher id's
class PitchTypeModel(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=8):
        super(PitchTypeModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

model = PitchTypeModel(
    input_dim=X_train_t.shape[1],
    num_classes=len(y_le.classes_),
)

# ==========================
# 3. Training
# ==========================
#Best for multi-class classification where targets are class indices
criterion = nn.CrossEntropyLoss(weight=class_weights)  
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

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
# 4. Evaluation
# ==========================
#Going to implement a different form of accuracy 
model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == y_test_t).float().mean()
    print(f"Test Accuracy: {accuracy.item()*100:.2f}%")



# --- Extended evaluation ---
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from collections import Counter
import matplotlib.pyplot as plt

probs = F.softmax(predictions, dim=1).cpu().numpy() #converts the logits into probabilities across classes
y_true = y_test_t.cpu().numpy() #converts tensor into a CPU NumPy array
y_pred = np.argmax(probs, axis=1) #For each row, pick the index of the class with the highest probability
class_names = list(y_le.classes_) #Back into human readable (FB, SL, CU)

# Class distribution & majority baseline
counts = Counter(y_true) #Build a frequency table 
maj_class, maj_count = counts.most_common(1)[0]
print("Class distribution (label: count):", counts) #Prints raw counts mapping to inspect class imbalance
#prints 1. numeric index of the majority class , 2. human readable class name, 3. majority class baseline accuracy
print(f"Majority class index: {maj_class} ({class_names[maj_class]}), baseline accuracy: {maj_count/len(y_true):.3f}") 

# Confusion matrix
cm = confusion_matrix(y_true, y_pred) #2d numpy array, rows=true, cols=pred
print("Confusion matrix (rows=true, cols=pred):")
print(cm)

# Classification report (precision, recall, f1)
unique_labels = np.unique(y_true)
target_names = [class_names[i] for i in unique_labels]
print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names))

# Macro F1
print("Macro F1:", f1_score(y_true, y_pred, average="macro")) #Unweighted mean of f1 scores for each class, treating all classes equally

# Top-k accuracy (k=2 and k=3)
def top_k_accuracy(probs, y_true, k=2):
    topk = np.argsort(probs, axis=1)[:, -k:]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])

print("Top-2 accuracy:", top_k_accuracy(probs, y_true, k=2)) # __% of the time, the true pitch was in the top 2 of guesses
print("Top-3 accuracy:", top_k_accuracy(probs, y_true, k=3)) #top 3 of guesses

# ROC-AUC and PR-AUC (One-vs-Rest)
#Fastball vs not fastball for example
num_classes = probs.shape[1]
y_bin = label_binarize(y_true, classes=np.arange(num_classes))  # shape (N, C)
try:
    per_class_auc = roc_auc_score(y_bin, probs, average=None)
    print("Per-class ROC AUC:", per_class_auc)
    print("Macro ROC AUC:", roc_auc_score(y_bin, probs, average="macro"))
except Exception as e:
    print("ROC AUC could not be computed:", e)

# Per-class average precision (PR-AUC)
per_class_ap = []
for i in range(num_classes):
    try:
        ap = average_precision_score(y_bin[:, i], probs[:, i])
    except Exception:
        ap = float("nan")
    per_class_ap.append(ap)
print("Per-class Average Precision (PR-AUC):", per_class_ap)

# Optional: plot confusion matrix heatmap (requires seaborn)
try:
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()
except Exception:
    pass