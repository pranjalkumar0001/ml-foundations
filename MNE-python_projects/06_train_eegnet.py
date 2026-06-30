import torch
import torch.nn as nn
import torch.optim as optim
from eegnet_model import EEGNet
import numpy as np

X_raw = np.load("X_binary.npy")
y_raw = np.load("y_binary.npy") -2

X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor(y_raw, dtype= torch.long)

X = X.unsqueeze(1) # X.size is (45,1,64,801)

model = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(50):
    model.train()
    optimiser.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimiser.step()
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y).sum().item()
    accuracy = correct / y.size(0)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/50] | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

