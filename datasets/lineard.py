import numpy as np


def make_linear_dataset(n_samples=100):
    X = np.linspace(-1, 1, n_samples)
    noise = 0.1 * np.random.randn(n_samples)

    y = 2 * X + 1 + noise

    return X.reshape(-1, 1), y