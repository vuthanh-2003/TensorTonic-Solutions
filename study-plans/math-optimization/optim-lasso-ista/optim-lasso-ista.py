import numpy as np

def lasso_ista(X: list, y: list, lam: float, n_iters: int) -> list:
    """
    Returns the complete ISTA weight trajectory as a list of lists.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    n,d = X.shape
    trajectory = []
    w = np.zeros(d)
    eig_max = np.max(np.linalg.eigvalsh((X.T@X)/n))
    t = 1/eig_max
    trajectory.append(w.tolist())
    for k in range(n_iters):
        z = w - (t/n)*X.T@(X@w - y)
        w = np.sign(z)*np.maximum(np.abs(z) - lam*t,0)
        trajectory.append(w.tolist())
    return trajectory