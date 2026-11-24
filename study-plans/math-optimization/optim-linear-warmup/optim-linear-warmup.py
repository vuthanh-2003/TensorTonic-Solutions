import numpy as np

def linear_warmup(X: list, y: list, base_lr: float, warmup_epochs: int, total_epochs: int) -> dict:
    """
    Returns the learning-rate schedule and pre-update losses.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    n,d = X.shape
    losses = []
    lr_schedule = []
    w = np.zeros(d)
    for e in range(total_epochs):
        lr = base_lr*np.minimum((e+1)/warmup_epochs, 1)
        lr_schedule.append(lr)
        losses.append(np.mean((X@w-y)**2))
        g = (2/n)*X.T@(X@w-y)
        w -= lr*g
    return {"lr_schedule": lr_schedule, "losses": losses}