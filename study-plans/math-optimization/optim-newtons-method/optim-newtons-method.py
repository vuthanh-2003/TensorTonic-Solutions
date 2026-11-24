import numpy as np

def newtons_method(X: list, y: list, lr_gd: float, n_iters: int) -> dict:
    """
    Returns gradient-descent and Newton MSE curves in a dictionary.
    """
    X = np.array(X,dtype = float)
    y = np.array(y,dtype = float)
    X = np.hstack([X, np.ones((X.shape[0],1))])
    n,d = X.shape
    def mse(w):
        r = X @ w - y
        return float(np.mean(r ** 2))

    H = (2.0 / n) * (X.T @ X)

    w_gd = np.zeros(d)
    w_nt = np.zeros(d)
    gd_losses = [mse(w_gd)]
    newton_losses = [mse(w_nt)]

    for _ in range(n_iters):
        g = (2.0 / n) * (X.T @ (X @ w_gd - y))
        w_gd = w_gd - lr_gd * g
        gd_losses.append(mse(w_gd))
        g = (2.0 / n) * (X.T @ (X @ w_nt - y))
        delta = np.linalg.solve(H, g)
        w_nt = w_nt - delta
        newton_losses.append(mse(w_nt))

    return {"gd_losses": gd_losses, "newton_losses": newton_losses}
    