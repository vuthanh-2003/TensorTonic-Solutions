import numpy as np

def gradient_clipping(X: list, y: list, lr: float, clip_value: float, clip_norm: float, n_epochs: int) -> dict:
    """
    Returns unclipped, value-clipped, and norm-clipped MSE curves.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape

    def mse(w, b):
        return float(np.mean((X @ w + b - y) ** 2))

    def grads(w, b):
        r = X @ w + b - y
        return (2.0 / n) * (X.T @ r), (2.0 / n) * np.sum(r)

    def run(mode):
        w = np.zeros(d)
        b = 0.0
        losses = []
        for _ in range(n_epochs):
            g_w, g_b = grads(w, b)
            if mode == "value":
                g_w = np.clip(g_w, -clip_value, clip_value)
                g_b = float(np.clip(g_b, -clip_value, clip_value))
            elif mode == "norm":
                g = np.append(g_w, g_b)
                norm = np.linalg.norm(g)
                if norm > clip_norm:
                    g = g * (clip_norm / norm)
                g_w, g_b = g[:-1], float(g[-1])
            w = w - lr * g_w
            b = b - lr * g_b
            losses.append(mse(w, b))
        return losses

    return {
        "unclipped_losses": run("none"),
        "clip_value_losses": run("value"),
        "clip_norm_losses": run("norm"),
    }