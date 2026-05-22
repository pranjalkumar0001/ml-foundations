# ==========================================
# Week 1: The Final Foundations
# Task: XOR Neural Network from Scratch in NumPy
# Author: Pranjal Kumar
# Date: May 22, 2026
# ==========================================

import numpy as np

# 1. Define the Dataset (XOR logic)
# X shape: (4, 2)
# Y shape: (4, 1)

# 2. Define the Activation Function
# Create a sigmoid(x) function. 

# 3. Initialize Parameters
# Set random seed for reproducibility (e.g., np.random.seed(42))
# W1 shape: (2, 3) - Initialize using np.random.randn
# b1 shape: (1, 3) - Initialize to zeros
# W2 shape: (3, 1) - Initialize using np.random.randn
# b2 shape: (1, 1) - Initialize to zeros

# Set hyperparameters (epochs = 10000, learning_rate = 0.1 or 1.0)

# 4. Training Loop
# for epoch in range(epochs):

    # --- FORWARD PASS ---
    # Compute Z1 = X @ W1 + b1
    # Compute A1 = sigmoid(Z1)
    # Compute Z2 = A1 @ W2 + b2
    # Compute Y_hat = sigmoid(Z2)
    
    # --- LOSS COMPUTATION ---
    # Calculate MSE Loss: np.mean((Y_hat - Y) ** 2)
    # (Optional: print the loss every 1000 epochs to monitor convergence)
    
    # --- BACKWARD PASS ---
    # Compute dZ2 = Y_hat - Y  
    # Compute dW2 = A1.T @ dZ2
    # Compute db2 = sum of dZ2 over the batch dimension (axis=0, keepdims=True)
    
    # Compute dZ1 = (dZ2 @ W2.T) * (A1 * (1 - A1))  (Note: * means element-wise multiplication here)
    # Compute dW1 = X.T @ dZ1
    # Compute db1 = sum of dZ1 over the batch dimension (axis=0, keepdims=True)
    
    # --- WEIGHT UPDATE ---
    # W2 = W2 - learning_rate * dW2
    # b2 = b2 - learning_rate * db2
    # W1 = W1 - learning_rate * dW1
    # b1 = b1 - learning_rate * db1

# 5. Final Output Test
# Print the final predictions (Y_hat) to verify they approximate [[0], [1], [1], [0]]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def back_prop(X, Y):
    w1 = np.random.randn(2,3)
    b1 = np.zeros((1,3))
    w2 = np.random.randn(3,1)
    b2 = np.zeros((1,1))
    learning_rate = 10
    for epoch in range (10000):
        z1 = X@w1 + b1
        a1 = sigmoid(z1)
        z2 = a1@w2 + b2
        predicted = sigmoid(z2)
        dz2 = (predicted-Y)*predicted*(1-predicted)
        dw2 = a1.T@dz2
        dz1 = dz2@(w2.T)*(a1*(1-a1))
        dw1 = X.T @ dz1
        db2 = np.sum(dz2, axis=0, keepdims=True)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        w1 -= dw1*learning_rate
        b1 -= db1*learning_rate
        w2 -= dw2*learning_rate
        b2 -= db2*learning_rate
        if (epoch % 1000 == 0):
            print(f"mse_loss in epoch:{epoch} is {np.mean((predicted-Y)**2)}")
    print(f"the values of w1={w1}, b1={b1}, w2={w2}, b2={b2}")

X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

Y = np.array([[0], 
              [1], 
              [1], 
              [0]])

back_prop(X,Y)