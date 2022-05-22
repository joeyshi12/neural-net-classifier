from typing import List
import numpy as np


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights, ())])


def unflatten_weights(weights_flat: np.ndarray, layer_sizes: List[int]):
    weights = []
    counter = 0
    for i in range(len(layer_sizes) - 1):
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        b_size = layer_sizes[i + 1]
        W = np.reshape(weights_flat[counter:counter + W_size], (layer_sizes[i + 1], layer_sizes[i]))
        counter += W_size
        b = weights_flat[counter:counter + b_size][None]
        counter += b_size
        weights.append((W, b))
    return weights


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)
