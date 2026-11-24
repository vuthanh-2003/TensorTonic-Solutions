import numpy as np

def adamw_compare(X: list, y: list, lr: float, beta1: float, beta2: float, weight_decay: float, n_epochs: int) -> dict:
    """
    Returns Adam with L2 and AdamW MSE loss curves in a dictionary.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n, d = X.shape
    eps = 1e-8

    # Initialize independent weights
    w_adam = np.zeros(d)
    w_adamw = np.zeros(d)

    # Initialize Adam L2 moments
    m_adam = np.zeros(d)
    v_adam = np.zeros(d)

    # Initialize AdamW moments
    m_adamw = np.zeros(d)
    v_adamw = np.zeros(d)

    adam_l2_losses = []
    adamw_losses = []

    for t in range(1, n_epochs + 1):
        pred = X @ w_adam
        loss = np.mean((pred - y) ** 2)
        adam_l2_losses.append(float(loss))
        grad = (2 / n) * X.T @ (pred - y)
        grad_l2 = grad + weight_decay * w_adam
        m_adam = beta1 * m_adam + (1 - beta1) * grad_l2
        v_adam = beta2 * v_adam + (1 - beta2) * (grad_l2 ** 2)
        m_hat = m_adam / (1 - beta1 ** t)
        v_hat = v_adam / (1 - beta2 ** t)
        w_adam = w_adam - lr * m_hat / (np.sqrt(v_hat) + eps)
        pred_w = X @ w_adamw
        loss_w = np.mean((pred_w - y) ** 2)
        adamw_losses.append(float(loss_w))

        grad_w = (2 / n) * X.T @ (pred_w - y)

        m_adamw = beta1 * m_adamw + (1 - beta1) * grad_w
        v_adamw = beta2 * v_adamw + (1 - beta2) * (grad_w ** 2)
        m_hat_w = m_adamw / (1 - beta1 ** t)
        v_hat_w = v_adamw / (1 - beta2 ** t)
        w_adamw = (
            w_adamw
            - lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
            - lr * weight_decay * w_adamw
        )

    return {
        "adam_l2_losses": adam_l2_losses,
        "adamw_losses": adamw_losses
    }
