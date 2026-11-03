
import numpy as np
import matplotlib.pyplot as plt
from math import erf

x = np.linspace(-5, 5, 1000)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def prelu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * x)


def rrelu(x, lower=1/8, upper=1/3):
    alpha = (lower + upper) / 2
    return np.where(x > 0, x, alpha * x)


def elu(x, alpha=1):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def selu(x):
    alpha = 1.67326
    scale = 1.0507
    return scale * np.where(
        x > 0,
        x,
        alpha * (np.exp(x) - 1)
    )


def softplus(x):
    return np.log1p(np.exp(x))


def softsign(x):
    return x / (1 + np.abs(x))


def swish(x):
    return x * sigmoid(x)


def hard_swish(x):
    return x * np.clip((x + 3) / 6, 0, 1)


def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))


def gelu(x):
    return 0.5 * x * (
        1 + np.vectorize(erf)(x / np.sqrt(2))
    )


def hard_sigmoid(x):
    return np.clip((x + 3) / 6, 0, 1)


def hard_tanh(x):
    return np.clip(x, -1, 1)


def tanh_shrink(x):
    return x - np.tanh(x)


def soft_shrink(x, lambd=0.5):
    return np.where(
        x > lambd,
        x - lambd,
        np.where(x < -lambd, x + lambd, 0)
    )


def hard_shrink(x, lambd=0.5):
    return np.where(np.abs(x) > lambd, x, 0)


def linear(x):
    return x


functions = [
    ("Sigmoid", sigmoid),
    ("Tanh", tanh),
    ("ReLU", relu),
    ("Leaky ReLU", leaky_relu),
    ("PReLU", prelu),
    ("RReLU", rrelu),
    ("ELU", elu),
    ("SELU", selu),
    ("Softplus", softplus),
    ("Softsign", softsign),
    ("Swish", swish),
    ("Hard Swish", hard_swish),
    ("Mish", mish),
    ("GELU", gelu),
    ("Hard Sigmoid", hard_sigmoid),
    ("Hard Tanh", hard_tanh),
    ("TanhShrink", tanh_shrink),
    ("SoftShrink", soft_shrink),
    ("HardShrink", hard_shrink),
    ("Linear", linear)
]

rows = 4
cols = 5

plt.figure(figsize=(15, 22))

for i, (name, func) in enumerate(functions):
    plt.subplot(rows, cols, i + 1)

    plt.plot(x, func(x), linewidth=2)

    plt.title(name, fontsize=12, pad=10)
    plt.xlabel("Input", fontsize=8)
    plt.ylabel("Output", fontsize=8)

    plt.grid(True)

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)

plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=3.5)

plt.subplots_adjust(hspace=0.7, wspace=0.5)

plt.show()
