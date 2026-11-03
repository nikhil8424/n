import numpy as np

X = np.array([
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
], dtype=float)

Y = np.array([
    [8],
    [11],
    [14],
    [17],
    [20]
], dtype=float)

W1 = 0.5
W2 = 0.3
b = 0.2

learning_rate = 0.01
epochs = 10000

print("Initial Weights")
print("W1 =", W1)
print("W2 =", W2)
print("Bias =", b)
print("-" * 50)

for epoch in range(epochs):
    total_error = 0

    for i in range(len(X)):
        x1 = X[i][0]
        x2 = X[i][1]
        target = Y[i][0]

        net = (x1 * W1) + (x2 * W2) + b
        output = net

        error = target - output
        total_error += error ** 2

        dW1 = -error * x1
        dW2 = -error * x2
        dB = -error

        W1 = W1 - learning_rate * dW1
        W2 = W2 - learning_rate * dW2
        b = b - learning_rate * dB

    mse = total_error / len(X)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} MSE = {mse:.6f}")

    if mse < 0.00001:
        print("\nTraining Completed")
        print("Epoch =", epoch)
        break

print("\nFinal Parameters")
print("W1 =", round(W1, 4))
print("W2 =", round(W2, 4))
print("Bias =", round(b, 4))

print("\nPredictions")
print("-" * 50)

for i in range(len(X)):
    x1 = X[i][0]
    x2 = X[i][1]

    predicted = (x1 * W1) + (x2 * W2) + b

    print(
        "Input =", X[i],
        "Actual =", Y[i][0],
        "Predicted =", round(predicted, 3),
        "Error =", round(Y[i][0] - predicted, 5)
    )
