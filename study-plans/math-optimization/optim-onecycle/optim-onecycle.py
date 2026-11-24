import numpy as np

def onecycle_train(X: list, y: list, max_lr: float, div_factor: float, final_div: float, pct_start: float, total_epochs: int) -> dict:
    """
    Returns the one-cycle schedule, losses, initial rate, and minimum rate.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    n,d = X.shape
    initial_lr = max_lr/div_factor
    min_lr =initial_lr/final_div
    peak_index = int(total_epochs*pct_start)
    peak_index = max(1, min(peak_index, total_epochs - 2))
    w = np.zeros(d)
    lr_schedule = []
    losses = []

    for epoch in range(total_epochs):
        if epoch <= peak_index:
            progress = epoch / peak_index

            lr = initial_lr + (max_lr - initial_lr) / 2 * (
                1 - np.cos(np.pi * progress)
            )
        else:
            progress = (epoch - peak_index) / (
                total_epochs - 1 - peak_index
            )

            lr = min_lr + (max_lr - min_lr) / 2 * (
                1 + np.cos(np.pi * progress)
            )
        lr_schedule.append(float(lr))

        pred = X @ w
        loss = np.mean((pred - y) ** 2)
        losses.append(float(loss))
        grad = (2 / n) * X.T @ (pred - y)
        w = w - lr * grad

    return {
        "lr_schedule": lr_schedule,
        "losses": losses,
        "initial_lr": float(initial_lr),
        "min_lr": float(min_lr)
    }