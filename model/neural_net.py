import numpy as np
from sklearn.preprocessing import LabelBinarizer


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights, ())])


def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
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


class NNClassifier:
    def __init__(self, hidden_layer_sizes=[100], alpha=0.001, lammy=1, epochs=10, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.lammy = lammy
        self.epochs = epochs
        self.verbose = verbose

    def fun_obj(self, weights_flat, X, Y):
        weights = unflatten_weights(weights_flat, self.layer_sizes)
        activations = [X]
        activation_derivs = list()
        for W, b in weights:
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
            activations.append(X)
            activation_derivs.append(X * (1 - X))

        f = np.sum(-Z[Y.astype(bool)] + np.log(np.sum(np.exp(Z), axis=1)))
        grad = np.exp(Z) / np.sum(np.exp(Z), axis=1)[:, None] - Y
        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)
        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            W, b = weights[i]
            grad = grad @ W * activation_derivs[i - 1]
            grad_W = grad.T @ activations[i - 1]
            grad_b = np.sum(grad, axis=0)
            g = [(grad_W, grad_b)] + g
        g = flatten_weights(g)

        f += 0.5 * self.lammy * np.sum(weights_flat ** 2)
        g += self.lammy * weights_flat
        return f, g

    def fit(self, X, y):
        Y = LabelBinarizer().fit_transform(y)
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [Y.shape[1]]
        size = 0
        for i in range(len(self.layer_sizes) - 1):
            size += self.layer_sizes[i + 1] * self.layer_sizes[i] + self.layer_sizes[i + 1]
        weights_flat = 0.01 * np.random.randn(size)

        num_batches = 100
        for epoch in range(self.epochs):
            reordered_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
            for i in range(num_batches):
                batch = reordered_indices[i * X.shape[0] // num_batches: (i + 1) * X.shape[0] // num_batches]
                f, g = self.fun_obj(weights_flat, X[batch], Y[batch])
                weights_flat -= self.alpha * g
            if self.verbose:
                print("loss:", f)

        self.weights = unflatten_weights(weights_flat, self.layer_sizes)

    def predict(self, X):
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
        return np.argmax(Z, axis=1)
