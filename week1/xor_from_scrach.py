import numpy as np

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
    print(f"the final predicted values are {predicted}")

X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

Y = np.array([[0], 
              [1], 
              [1], 
              [0]])

back_prop(X,Y)