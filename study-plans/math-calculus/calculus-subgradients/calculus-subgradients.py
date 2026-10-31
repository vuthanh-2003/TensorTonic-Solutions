import numpy as np

def subgradient_analysis(x_points: list, w_init: float, lr: float, n_iters: int) -> dict:
    """
    Returns selected subgradients and the descent trajectory.
    """
    abs_subgrad = [float(np.sign(x)) for x in x_points]
    relu_subgrad = [1.0 if x > 0 else 0.0 for x in x_points]
    w = float(w_init)
    w_trajectory = [w]
    for _ in range(n_iters):
        g = float(np.sign(w - 3)) + w
        w = w -lr*g
        w_trajectory.append(w)
    return {
        "abs_subgrad": abs_subgrad,
        "relu_subgrad": relu_subgrad,
        "w_trajectory": w_trajectory,
        "w_final": w_trajectory[-1]
    }