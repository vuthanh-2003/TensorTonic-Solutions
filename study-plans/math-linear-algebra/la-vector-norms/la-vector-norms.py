import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.array(v)
    norm = np.zeros(3)
    norm[0:1] = np.sum(abs(v))
    norm[1:2] = np.linalg.norm(v)
    norm[2:3] = max(abs(v))
    return norm