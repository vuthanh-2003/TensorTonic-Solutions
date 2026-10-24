import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype = np.float64)
    y = np.array(y, dtype = np.float64).reshape(-1,1)
    n,d = X.shape
    weights = np.zeros((d+1,1))
    identity = np.eye(d+1)
    columns_one = np.ones((n,1))
    X = np.hstack((columns_one, X))
    identity[0,0] = 0
    for epoch in range(epochs):
        gradient = -(2 / n) * X.T @ (y - X @ weights) + 2 * alpha * identity @ weights
        weights -= lr * gradient
    return (np.round(weights[1:], 4).flatten().tolist(), np.round(weights[0,0],4))
