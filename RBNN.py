
import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

T = np.array([0, 1, 1, 0])

C = X.copy()

sigma = 1.0

H = []

for x in X:
    d = np.sum((x - C) ** 2, axis=1)
    h = np.exp(-d / (2 * sigma ** 2))
    H.append(h)

H = np.array(H)

W = np.linalg.pinv(H) @ T

print("Hidden Layer Matrix:")
print(np.round(H, 3))

print("\nCalculated Output Weights:")
print(np.round(W, 4))

print("\nRBF NEURAL NETWORK XOR RESULTS")
print("--------------------------------")

for x in X:
    d = np.sum((x - C) ** 2, axis=1)
    H_test = np.exp(-d / (2 * sigma ** 2))
    Y = np.sum(H_test * W)

    if Y > 0.5:
        output = 1
    else:
        output = 0

    print("\nInput:", x)
    print("Distance Squared:", d)
    print("Hidden Layer Outputs:", np.round(H_test, 3))
    print("Network Output:", round(Y, 4))
    print("Predicted XOR Output:", output)

print("\n--------------------------------")
print("Training Targets:", T)
print("Predicted Outputs:")

predictions = []

for x in X:
    d = np.sum((x - C) ** 2, axis=1)
    H_test = np.exp(-d / (2 * sigma ** 2))
    Y = np.sum(H_test * W)

    predictions.append(1 if Y > 0.5 else 0)

print(predictions)

if np.array_equal(predictions, T):
    print("\nXOR Classification Successful!")
else:
    print("\nXOR Classification Failed!")
