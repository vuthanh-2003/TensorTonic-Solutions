import numpy as np

def optimizer_showdown(X: list, y: list, lr: float, beta1: float, beta2: float, epsilon: float, n_epochs: int, batch_size: int) -> dict:
    """
    Returns SGD, momentum, and Adam BCE curves in a dictionary.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(w, b):
        p = sigmoid(X @ w + b)
        p = np.clip(p, 1e-15, 1 - 1e-15)

        return -np.mean(
            y * np.log(p) + (1 - y) * np.log(1 - p)
        )
    losses = {
        "sgd_losses": [],
        "momentum_losses": [],
        "adam_losses": []
    }
    for optimizer in losses:

        w = np.zeros(n_features)
        b = 0.0

        vw = np.zeros(n_features)
        vb = 0.0
        mw = np.zeros(n_features)
        mb = 0.0

        sw = np.zeros(n_features)
        sb = 0.0
        t = 0

        for epoch in range(n_epochs):

            for start in range(0, n_samples, batch_size):

                end = min(start + batch_size, n_samples)

                X_batch = X[start:end]
                y_batch = y[start:end]

                r = len(y_batch)
                p = sigmoid(X_batch @ w + b)

                gw = X_batch.T @ (p - y_batch) / r
                gb = np.sum(p - y_batch) / r

                if optimizer == "sgd_losses":

                    w -= lr * gw
                    b -= lr * gb

                elif optimizer == "momentum_losses":

                    vw = beta1 * vw + gw
                    vb = beta1 * vb + gb

                    w -= lr * vw
                    b -= lr * vb

                else:

                    t += 1

                    mw = beta1 * mw + (1 - beta1) * gw
                    mb = beta1 * mb + (1 - beta1) * gb

                    sw = beta2 * sw + (1 - beta2) * (gw ** 2)
                    sb = beta2 * sb + (1 - beta2) * (gb ** 2)

                    mw_hat = mw / (1 - beta1 ** t)
                    mb_hat = mb / (1 - beta1 ** t)

                    sw_hat = sw / (1 - beta2 ** t)
                    sb_hat = sb / (1 - beta2 ** t)

                    w -= lr * mw_hat / (np.sqrt(sw_hat) + epsilon)
                    b -= lr * mb_hat / (np.sqrt(sb_hat) + epsilon)

            losses[optimizer].append(compute_loss(w, b))

    return losses
