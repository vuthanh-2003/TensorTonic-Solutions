import numpy as np

def learning_rate_sweep(X: list, y: list, learning_rates: list, n_epochs: int) -> list:
    """
    Returns one pre-update MSE loss curve per learning rate.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    n,d = X.shape
    losses = []
    for lr in learning_rates:
        w = np.zeros(d)
        loss = []
        for epoch in range(n_epochs):
            r = X@w - y
            mse = np.mean(r**2)
            loss.append(mse)
            g = (2/n)*X.T@(X@w - y)
            w -= lr*g
        losses.append(loss)
    return losses