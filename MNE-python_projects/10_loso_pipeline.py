import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mne
from eegnet_model import EEGNet

torch.manual_seed(42)
np.random.seed(42)

def get_subject_data(subjects):
    runs = [4, 8, 12]
    X_list, y_list = [], []
    for sub in subjects:
        raw_fnames = mne.datasets.eegbci.load_data(sub, runs)
        raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
        for raw in raws:
            mne.datasets.eegbci.standardize(raw)
        raw = mne.concatenate_raws(raws)
        events, _ = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=5.0, baseline=None, preload=True)
        
        X = epochs.get_data()
        y = epochs.events[:, -1]
        mask = (y == 2) | (y == 3)
        
        X_list.append(X[mask])
        y_list.append(y[mask])
        
    # Apply the Microvolt Scale Fix (* 1e6) and PyTorch Label Shift (- 2) immediately
    return np.concatenate(X_list, axis=0) * 1e6, np.concatenate(y_list, axis=0) - 2

print("Downloading Training Vault (Subjects 1-4)...")
X_train_raw, y_train_raw = get_subject_data([1, 2, 3, 4])

print("Downloading Testing Vault (Subject 5)...")
X_test_raw, y_test_raw = get_subject_data([5])

X_train = torch.tensor(X_train_raw, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train_raw, dtype=torch.long)

X_test = torch.tensor(X_test_raw, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_raw, dtype=torch.long)

# --- 4. THE MINI-BATCH GENERATOR ---
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


model = EEGNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# --- TRAINING PHASE (Strictly Subjects 1-4) ---
print("\nInitiating Training Loop...")
epochs = 50
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# --- VALIDATION PHASE (Strictly Subject 5) ---
model.eval() 
with torch.no_grad(): 
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test).sum().item()
    test_accuracy = correct / y_test.size(0)

print(f"\n--- THE TRUE BENCHMARK ---")
print(f"Subject 5 (Complete Stranger) Accuracy: {test_accuracy * 100:.2f}%")