def mse_loss(predicted, target):
    n = len(predicted)
    total_loss = sum((p-t)**2 for p,t in zip(predicted, target))
    return total_loss/n

#for gradient descent we need to find weights for which the mse_loss is minimum. we use 0.1 as the step of change of weights.
#assuming a basic neuron with 1 weight and bias. y = w*x+b.

def gradient_descent(input, target):
    w = 0
    b = 0
    learning_rate = 0.1
    # for _ in range(100):
    #     for i in range(len(input)):
    #         predicted = [w*x + b for x in input]
    #         p = predicted[i]
    #         t = target[i]
    #         dw = 2*input[i]*(p-t)
    #         db = 2*(p-t)
    #         w = w - dw*learning_rate
    #         b = b - db*learning_rate
    #     print(mse_loss(predicted, target))
    #more neater way with higher epochs
    for epoch in range(1000):
        predicted = [w*x + b for x in input]
        dw = sum(2*input[i]*(predicted[i]-target[i]) for i in range(len(input))) / len(input)
        db = sum(2*(predicted[i]-target[i]) for i in range(len(input))) / len(input)
        w = w - dw*learning_rate
        b = b - db*learning_rate
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: mse_loss = {mse_loss(predicted, target)}")
    print(f"w={w} and b={b}")


input = [1,2,3,4]
target = [5,7,9,11]
gradient_descent(input, target)