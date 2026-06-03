import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

def load_and_window_motor_data(csv_path, seq_length=100, train_split=0.8):
    print("Loading data... (this might take a few seconds)")
    
    input_cols = ['u_d', 'u_q', 'i_d', 'i_q', 'coolant']
    target_col = 'pm'
    
    raw_X_list = []
    raw_Y_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        profile_idx = header.index('profile_id')
        x_indices = [header.index(col) for col in input_cols]
        y_idx = header.index(target_col)
        
        for row in reader:
            # Filter specifically for profile 4
            if int(row[profile_idx]) == 4:
                raw_X_list.append([float(row[i]) for i in x_indices])
                raw_Y_list.append([float(row[y_idx])])
                
    raw_X = np.array(raw_X_list)
    raw_Y = np.array(raw_Y_list)
    
    # 2. Manual Standardization (Mean=0, Variance=1)
    mean_X = np.mean(raw_X, axis=0)
    std_X = np.std(raw_X, axis=0)
    # The 1e-8 prevents catastrophic division by zero if a sensor breaks and reads 0 constantly
    scaled_X = (raw_X - mean_X) / (std_X + 1e-8) 
    
    mean_Y = np.mean(raw_Y, axis=0)
    std_Y = np.std(raw_Y, axis=0)
    scaled_Y = (raw_Y - mean_Y) / (std_Y + 1e-8)
    
    # 3. Create the 3D Sliding Windows [Batch, Time, Features]
    X, Y = [], []
    for i in range(len(scaled_X) - seq_length):
        X.append(scaled_X[i : i + seq_length])
        Y.append(scaled_Y[i + seq_length]) 
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    
    # 4. Strict Time-Series Split
    split_idx = int(train_split * len(X))
    
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    return X_train, Y_train, X_test, Y_test

# link to download the data: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature
# Provide your absolute path to the Downloads folder here
csv_location = '/Users/pranjalkumar/Downloads/measures_v2.csv'
X_train, Y_train, X_test, Y_test = load_and_window_motor_data(csv_location)

# print(f"Training Data: {X_train.shape} -> Targets: {Y_train.shape}")
# print(f"Testing Data:  {X_test.shape} -> Targets: {Y_test.shape}")


# training begins here
lstm = nn.LSTM(5, 64, batch_first=True)
head = nn.Linear(64, 1)
loss_fn = nn.MSELoss()
optimiser = optim.Adam(list(lstm.parameters())+ list(head.parameters()), lr=0.001)
for epoch in range(101):
    optimiser.zero_grad()
    _, (final_hidden, cell) = lstm(X_train)
    prediction = head(final_hidden.squeeze())
    loss = loss_fn(prediction, Y_train)
    loss.backward()
    optimiser.step()
    if epoch % 10 == 0:
        print(f"for epoch = {epoch}, mse loss is {loss.item()}")

#testing begins here
with torch.no_grad():
    _, (test_hidden, test_cell) = lstm(X_test)
    test_predictions = head(test_hidden.squeeze())
    test_loss = loss_fn(test_predictions, Y_test)
    print(f"error in the test case is {test_loss.item()}")