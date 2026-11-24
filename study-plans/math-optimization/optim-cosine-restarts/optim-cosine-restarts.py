import numpy as np

def cosine_restarts(X: list, y: list, eta_max: float, eta_min: float, T_0: int, T_mult: int, total_epochs: int) -> dict:
    """
    Returns the cosine-restart schedule and pre-update losses.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n, d = X.shape
    w = np.zeros(d)
    T_i = T_0
    T_cur = 0
    lr_schedule = []
    losses = []

    for epoch in range(total_epochs):
        lr = eta_min + (eta_max - eta_min) / 2 * (
            1 + np.cos(np.pi * T_cur / T_i)
        )

        lr_schedule.append(float(lr))
        pred = X @ w
        loss = np.mean((pred - y) ** 2)
        losses.append(float(loss))

        grad = (2 / n) * X.T @ (pred - y)
        w = w - lr * grad
        T_cur += 1
        if T_cur == T_i:
            T_cur = 0
            T_i = T_i * T_mult

    return {
        "lr_schedule": lr_schedule,
        "losses": losses
    }
