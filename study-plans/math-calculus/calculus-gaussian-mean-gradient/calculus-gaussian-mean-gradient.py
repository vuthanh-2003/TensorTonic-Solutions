import numpy as np

def gaussian_log_mean_gradient(x: list, mu: list, Sigma: list) -> list:
    """
    Returns the log-density gradient with respect to the mean.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)

    residual = x - mu

    g = np.linalg.solve(Sigma, residual)

    return [round(float(v), 6) for v in g]
