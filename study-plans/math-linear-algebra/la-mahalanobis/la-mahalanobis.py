import numpy as np

def mahalanobis_distance(x, mean, cov):
    """
    Returns: float, the Mahalanobis distance from x to the distribution.
    """
    x = np.array(x)
    mean = np.array(mean)
    cov = np.array(cov)
    cov_pinv = np.linalg.pinv(cov)
    d = np.sqrt((x - mean).T@cov_pinv@(x-mean))
    return d