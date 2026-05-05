import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    X = np.array(X, dtype = np.float64)
    W = np.array(W, dtype = np.float64)
    Y = X@W
    norms = np.linalg.norm(Y, axis = 1)
    gate = (norms >= threshold).astype(Y.dtype)
    return Y*gate[:,np.newaxis]