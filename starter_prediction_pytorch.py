#install these libraries
#pip/pip3 install pybaseball pandas torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from pybaseball import statcast
import pandas as pd

#Using 5 days of data
df = statcast(start_dt='2025-09-01', end_dt='2025-09-05')

#Clean data
# Convert columns to numeric, forcing invalid values to NaN
for col in ['release_speed', 'release_spin_rate']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values after conversion
df = df.dropna(subset=['release_speed', 'release_spin_rate', 'description'])

#1 if strike, 0 if ball
df['target'] = df['description'].apply(lambda x: 1 if isinstance(x, str) and 'strike' in x.lower() else 0)

#Ensuring there are only float types
X_np = df[['release_speed', 'release_spin_rate']].astype(float).values
y_np = df['target'].astype(float).values

#Preparing Tensors
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

# Split into train/test. 80% to train, 20% to test
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#Neural Network with 1 hidden layer
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 8) #Fully connected layer, 2 inputs -> 8 hidden units
        self.relu = nn.ReLU() #ReLU activation function for non-linearity
        self.fc2 = nn.Linear(8, 1) #Fully connected layer 8 hidden inputs -> 1 output
        self.sigmoid = nn.Sigmoid () #Sigmoid activation outputs probability between 0 and 1

    def forward(self, x):
        x = self.relu(self.fc1(x)) #first layer + ReLU
        x = self.sigmoid(self.fc2(x)) #Second layer + sigmoid
        return x
    
#Initializing model, loss, and optimizer
model = SimpleNet()
criterion = nn.BCELoss()  #Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01) #Adam optimizer with learning rate 0.01

#Training loop
for epoch in range(100):
    optimizer.zero_grad() #Resetting gradients
    outputs = model(X_train) #forward pass , get perdictions
    loss = criterion(outputs, y_train) #computes loss
    loss.backward() #compute gradients
    optimizer.step() #updates weights based on gradients
 
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')


#Evaluate
with torch.no_grad(): #disable gradient calculation to save memory
    preds = model(X_test) #Forward pass on test set
    preds = (preds > 0.5).float() #Converts probabilities to 0 or 1
    acc = (preds.eq(y_test).sum() / y_test.shape[0]).item() #computing accuracy
    print(f'Accuacy: {acc:.2f}')

torch.save(model.state_dict(), 'starter_prediction.pt')