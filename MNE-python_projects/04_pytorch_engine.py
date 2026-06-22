import torch
import numpy as np
x_raw = np.load('X_binary.npy')
y_raw = np.load('y_binary.npy')
#print(y_raw)
y_raw -= 2
X = torch.tensor(x_raw, dtype=torch.float32)
y = torch.tensor(y_raw,dtype=torch.long)

print(f"X Tensor Shape: {X.shape}")
print(f"y Tensor Shape: {y.shape}")