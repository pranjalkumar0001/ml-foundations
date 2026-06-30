import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from eegnet_model import EEGNet
import numpy as np

X_raw = np.load("X_binary.npy")
y_raw = np.load("y_binary.npy") -2

X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor(y_raw, dtype= torch.long)

X = X.unsqueeze(1) # X.size is (45,1,64,801)
# 1. The Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Engine Setup
model = EEGNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3. The Training Loop (Trained ONLY on X_train)
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# 4. The Validation Phase (The True Benchmark)
model.eval() # Freeze the architecture
with torch.no_grad(): # Shut off the learning engine
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test).sum().item()
    test_accuracy = correct / y_test.size(0)

print(f"Real-World Test Accuracy: {test_accuracy * 100:.2f}%")