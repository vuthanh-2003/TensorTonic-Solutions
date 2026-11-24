import numpy as np

def lbfgs(X: list, y: list, lr_gd: float, m: int, n_iters: int) -> dict:
    """
    Returns gradient-descent and L-BFGS MSE curves in a dictionary.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape

    def loss(w):
        r = X @ w - y
        return float(np.mean(r ** 2))

    def grad(w):
        return (2.0 / n) * (X.T @ (X @ w - y))

    w_gd = np.zeros(d, dtype=np.float64)
    gd_losses = [loss(w_gd)]
    for _ in range(n_iters):
        w_gd = w_gd - lr_gd * grad(w_gd)
        gd_losses.append(loss(w_gd))

    def two_loop(g, S, Z):
        q = g.copy()
        k = len(S)
        rhos = [1.0 / (Z[i] @ S[i] + 1e-10) for i in range(k)]
        alphas = [0.0] * k
        for i in range(k - 1, -1, -1):
            alphas[i] = rhos[i] * (S[i] @ q)
            q = q - alphas[i] * Z[i]
        if k > 0:
            gamma = (S[-1] @ Z[-1]) / (Z[-1] @ Z[-1] + 1e-10)
        else:
            gamma = 1.0
        r = gamma * q
        for i in range(k):
            beta = rhos[i] * (Z[i] @ r)
            r = r + S[i] * (alphas[i] - beta)
        return -r

    def armijo(w, g, d_vec):
        c = 1e-4
        t = 1.0
        f0 = loss(w)
        slope = g @ d_vec
        for _ in range(20):
            if loss(w + t * d_vec) <= f0 + c * t * slope:
                break
            t *= 0.5
        return t

    w = np.zeros(d, dtype=np.float64)
    lbfgs_losses = [loss(w)]
    S, Z = [], []
    g = grad(w)
    for _ in range(n_iters):
        d_vec = two_loop(g, S, Z)
        t = armijo(w, g, d_vec)
        w_new = w + t * d_vec
        g_new = grad(w_new)

        S.append(w_new - w)
        Z.append(g_new - g)
        if len(S) > m:
            S.pop(0)
            Z.pop(0)

        w, g = w_new, g_new
        lbfgs_losses.append(loss(w))

    return {"gd_losses": gd_losses, "lbfgs_losses": lbfgs_losses}