import numpy as np

def rmsprop(X: list, y: list, lr: float, decay: float, n_epochs: int) -> dict:
    """
    Returns MSE losses and effective learning rates in a dictionary.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    n,d = X.shape
    E = np.zeros(d)
    w = np.zeros(d)
    losses = []
    eff_lrs = []
    for _ in range(n_epochs):
        r = X@w - y
        mse = np.mean(r**2)
        losses.append(mse)
        gradient = (2/n)*X.T@(X@w-y)
        E = decay*E + (1-decay)*(gradient*gradient)
        eff_lr = lr/(np.sqrt(E+1e-8))
        eff_lrs.append(eff_lr)
        w -= eff_lr*gradient
    return {"losses": losses, "effective_lrs": eff_lrs}