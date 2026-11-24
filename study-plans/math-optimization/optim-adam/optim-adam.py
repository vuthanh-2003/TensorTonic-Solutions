import numpy as np

def adam(X: list, y: list, lr: float, beta1: float, beta2: float, n_epochs: int) -> dict:
    """
    Returns pre-update MSE losses and final weights in a dictionary.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    n,d = X.shape
    m = np.zeros(d)
    v = np.zeros(d)
    losses = []
    w = np.zeros(d)
    for t in range(1,n_epochs+1):
        r = X@w - y
        mse = np.mean(r**2)
        losses.append(mse)
        gradient = (2/n)*X.T@(X@w - y)
        m = beta1*m + (1-beta1)*gradient
        v = beta2*v + (1-beta2)*gradient**2
        m_hat = m/(1-beta1**t)
        v_hat = v/(1-beta2**t)
        w -= lr*m_hat/(np.sqrt(v_hat)+1e-8)
    return {"losses": losses, "final_weights": w.tolist()}