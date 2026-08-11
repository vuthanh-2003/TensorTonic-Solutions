import numpy as np

def outer_product(u, v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    u = np.array(u)
    u = u.reshape(-1,1)
    v = np.array(v)
    v = v.reshape(1,-1)
    return u@v