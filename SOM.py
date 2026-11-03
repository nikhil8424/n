
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("data.csv")

# Select numerical columns
data = df.select_dtypes(include=np.number)

# Remove missing values
data = data.dropna()

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(data)

# Create SOM
som = MiniSom(
    x=5,
    y=5,
    input_len=X.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)

# Initialize weights
som.random_weights_init(X)


som.train_random(X, 1000)

print("SOM Training Completed!")

# Display quantization error
print("Quantization Error:", som.quantization_error(X))

# Find winning neurons
for i, x in enumerate(X[:10]):
    print("Sample", i + 1, "Winner:", som.winner(x))

# Display U-Matrix
plt.figure(figsize=(7, 7))

plt.title("SOM U-Matrix")
plt.pcolor(som.distance_map().T, cmap="viridis")
plt.colorbar(label="Distance")

plt.show()

# Display Hit Map
plt.figure(figsize=(7, 7))

plt.title("SOM Hit Map")

frequencies = som.activation_response(X)

plt.pcolor(frequencies.T, cmap="Blues")
plt.colorbar(label="Number of Samples")

for x in range(5):
    for y in range(5):
        plt.text(
            x + 0.5,
            y + 0.5,
            int(frequencies[x, y]),
            ha="center",
            va="center"
        )

plt.show()
