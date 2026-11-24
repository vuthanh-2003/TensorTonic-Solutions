import numpy as np

def gradient_accumulation(X: list, y: list, lr: float, micro_batch: int, accum_steps: int, n_epochs: int) -> dict:
    """
    Returns both loss curves and final weight vectors in a dictionary.
    """
    X = np.array(X,dtype = float)
    y = np.array(y,dtype = float)
    n,d = X.shape
    w_no = np.zeros(d)
    w_acc = np.zeros(d)
    no_accum_losses = []
    accum_losses = []
    def gradient_batch(X_b, y_b,w):
        return 2*X_b.T@(X_b@w - y_b)
    for _ in range(n_epochs):
        for s in range(0, n, micro_batch):
            X_b, y_b = X[s:s + micro_batch], y[s:s + micro_batch]
            w_no = w_no - lr * gradient_batch(X_b, y_b, w_no) / len(y_b)
        grad_sum = np.zeros(d)
        count = 0
        steps = 0
        for s in range(0, n, micro_batch):
            X_b, y_b = X[s:s + micro_batch], y[s:s + micro_batch]
            grad_sum += gradient_batch(X_b, y_b, w_acc)
            count += len(y_b)
            steps += 1
            if steps == accum_steps:
                w_acc = w_acc - lr * grad_sum / count
                grad_sum = np.zeros(d)
                count = 0
                steps = 0
        if steps > 0:
            w_acc = w_acc - lr * grad_sum / count

        no_accum_losses.append(float(np.mean((X @ w_no - y) ** 2)))
        accum_losses.append(float(np.mean((X @ w_acc - y) ** 2)))

    return {
        "no_accum_losses": no_accum_losses,
        "accum_losses": accum_losses,
        "no_accum_weights": [float(v) for v in w_no],
        "accum_weights": [float(v) for v in w_acc],
    }
            
            
    