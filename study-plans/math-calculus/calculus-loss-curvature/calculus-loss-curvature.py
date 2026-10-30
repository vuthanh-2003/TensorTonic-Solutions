import numpy as np

def loss_curvature_analysis(y_hat: list, y: list, delta: float) -> dict:
    """
    Returns first and second derivatives for three losses.
    """
    y_hat = np.array(y_hat, dtype = np.float64)
    y = np.array(y, dtype = np.float64)
    dL_mse = 2*(y_hat-y)
    d2L_mse = np.full(dL_mse.shape, 2.0)
    dL_ce = -1*(y/y_hat) + (1-y)/(1-y_hat)
    d2L_ce = y/(y_hat**2) + (1-y)/(1-y_hat)**2
    error = y_hat - y
    abs_error = np.abs(error)

    dL_huber = np.where(
        abs_error <= delta,
        error,
        np.sign(error) * delta)

    d2L_huber = np.where(abs_error < delta, 1.0, 0.0)
    return {"mse": {"dL": dL_mse.flatten().tolist(), "d2L": d2L_mse.flatten().tolist()}, "ce": {"dL": dL_ce.flatten().tolist(), "d2L": d2L_ce.flatten().tolist()}, "huber": {"dL": dL_huber.flatten().tolist(), "d2L": d2L_huber.flatten().tolist()}}