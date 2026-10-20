import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype = np.float64)
    y = np.array(y, dtype = np.float64).reshape(-1,1)
    m,n = X.shape
    weights = np.zeros((n,1))
    bias = 0.0
    for epoch in range(epochs):
        error = X@weights + bias - y
        weights -= lr*((2/m)*X.T@error)
        bias -= lr*(2/m)*np.sum(error)
    return (weights.flatten().tolist(), bias)
