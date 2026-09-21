import numpy as np

def rbf_kernel_matrix(X: list, gamma: float) -> np.ndarray:
    """
    Returns the float64 pairwise RBF kernel matrix.
    """
    X = np.array(X, dtype = np.float64)
    d = np.sum(X**2, axis = 1)
    distance = d[:, None] + d[None,:] -2*X@X.T
    return np.exp(-1*gamma*distance)