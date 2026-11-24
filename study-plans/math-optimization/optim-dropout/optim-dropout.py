import numpy as np

def dropout_mlp(X_train: list, y_train: list, X_test: list, y_test: list, hidden_size: int, lr: float, dropout_rate: float, n_epochs: int, seed: int) -> list:
    """
    Returns one post-update test accuracy per epoch.
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float).reshape(-1, 1)
    X_test = np.array(X_test, dtype=float)
    y_test = np.array(y_test, dtype=float).reshape(-1, 1)

    n, d = X_train.shape

    np.random.seed(seed)
    W1 = np.random.randn(d, hidden_size) * np.sqrt(2.0 / d)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, 1))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    keep = 1.0 - dropout_rate
    accs = []

    for _ in range(n_epochs):
        # ---- forward (có dropout) ----
        z1 = X_train @ W1 + b1
        a1 = np.maximum(0, z1)
        mask = (np.random.rand(*a1.shape) > dropout_rate).astype(float)
        scale = mask / keep
        a1d = a1 * scale
        z2 = a1d @ W2 + b2
        y_hat = sigmoid(z2)

        # ---- backward (dùng lại mask) ----
        dz2 = (y_hat - y_train) / n
        dW2 = a1d.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1d = dz2 @ W2.T
        da1 = da1d * scale
        dz1 = da1 * (z1 > 0)
        dW1 = X_train.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # ---- đánh giá: không dropout ----
        h = np.maximum(0, X_test @ W1 + b1)
        pred = (sigmoid(h @ W2 + b2) >= 0.5).astype(float)
        accs.append(float(np.mean(pred == y_test)))

    return accs