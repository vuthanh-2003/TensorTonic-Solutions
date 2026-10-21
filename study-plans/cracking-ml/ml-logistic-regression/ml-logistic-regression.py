import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype = np.float64)
    y = np.array(y, dtype = np.float64).reshape(-1,1)
    n,d = X.shape
    weights = np.zeros((d,1))
    bias = 0.0
    for iter in range(n_iters):
        logit = X@weights + bias
        logit_clipped = np.clip(logit, -500, 500)
        p = 1/(1+np.exp(-1*logit_clipped))
        weights -= lr*(1/n)*X.T@(p-y)
        bias -= lr*(1/n)*np.sum(p-y)
    return (np.round(weights,decimals = 4).flatten().tolist(), round(float(bias),4))
