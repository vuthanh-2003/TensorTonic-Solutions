import numpy as np

def batch_gd_compare(X: list, y: list, batch_sizes: list, n_epochs: int, lr: float, seed: int) -> list:
    """
    Returns one post-epoch MSE loss curve per batch size.
    """
    X_arr = np.array(X, dtype = np.float64)
    y_arr = np.array(y, dtype = np.float64)
    n_samples, n_features = X_arr.shape

    all_loss_curves = []
    for b_size in batch_sizes:
        w = np.zeros(n_features, dtype = np.float64)
        rng = np.random.RandomState(seed)
        loss_curve = []
        for epoch in range(n_epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X_arr[indices]
            y_shuffled = y_arr[indices]
            for start_idx in range(0, n_samples, b_size):
                X_batch = X_shuffled[start_idx:start_idx+b_size]
                y_batch = y_shuffled[start_idx:start_idx+b_size]
                B = len(X_batch)
                gradient = (2.0/B)*(X_batch.T@(X_batch@w-y_batch))
                w -= lr*gradient
            full_predict = X_arr@w
            mse = float(np.mean((full_predict - y_arr)**2))
            loss_curve.append(mse)
        all_loss_curves.append(loss_curve)
    return all_loss_curves