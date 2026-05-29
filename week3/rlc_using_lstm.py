import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def get_circuit_transient_data(seq_length=50, total_samples=1000):
    """
    Generates deterministic, physical voltage data for an underdamped RLC circuit.
    Splits into 80% Train and 20% Test for validation.
    """
    t = np.linspace(0, 15, total_samples)
    voltage = np.exp(-0.2 * t) * np.cos(2 * np.pi * 0.75 * t)
    X, Y = [], []
    for i in range(len(voltage) - seq_length - 1):
        X.append(voltage[i : i + seq_length])    
        Y.append(voltage[i + seq_length])         
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1) 
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1) 
    
    # 3. TIME-SERIES SPLIT (80% Train, 20% Test)
    split_idx = int(0.8 * len(X))
    
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    return X_train, Y_train, X_test, Y_test
X_train, Y_train, X_test, Y_test = get_circuit_transient_data()


print(f"Training Data: {X_train.shape} -> Targets: {Y_train.shape}")
print(f"Testing Data:  {X_test.shape} -> Targets: {Y_test.shape}")
lstm_layer = nn.LSTM(1, 64, batch_first=True)
head = nn.Linear(64, 1)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(list(lstm_layer.parameters()) + list(head.parameters()), 0.001)
print("started training")
for epoch in range(151):
    optimizer.zero_grad()
    _, (final_hidden, cell) = lstm_layer(X_train)
    out = head(final_hidden.squeeze())
    loss = loss_fn(out, Y_train)
    loss.backward()
    optimizer.step()
    if epoch % 30 == 0:
        print(f"for epoch = {epoch}, the mse loss is {loss.item()}")

#testing overfitting via test cases
print("started testing")
with torch.no_grad():
    _, (test_hidden, test_cell) = lstm_layer(X_test)
    test_out = head(test_hidden.squeeze())
    test_loss = loss_fn(test_out, Y_test)

print(f"for test cases the loss is {test_loss}")

