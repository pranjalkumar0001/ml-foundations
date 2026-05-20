import math

def mse_loss(predicted, target):
    n = len(predicted)
    total_c = sum((p-t)**2 for p,t in zip(predicted, target))
    return total_c/n

#test
prediction = [0.55, 0.84, 0.99, 0.04]
target = [0, 1, 1, 0]
cost = mse_loss(prediction, target)
print(f"mse_loss is {cost}")
