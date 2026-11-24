import numpy as np

def adagrad(X: list, y: list, lr: float, n_epochs: int) -> dict:
    """
    Returns MSE losses and effective learning rates in a dictionary.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    n,d = X.shape
    w = np.zeros(d)
    G = np.zeros(d)
    eff_lrs = []
    losses = []
    for _ in range(n_epochs):
        r = X@w - y
        mse = np.mean(r**2)
        losses.append(mse)
        gradient = (2/n)*X.T@(X@w - y)
        G = G + gradient*gradient
        eff_lr = lr/(np.sqrt(G+1e-8))
        eff_lrs.append(eff_lr)
        w = w - eff_lr*gradient
    return {"losses": losses, "effective_lrs": eff_lrs}