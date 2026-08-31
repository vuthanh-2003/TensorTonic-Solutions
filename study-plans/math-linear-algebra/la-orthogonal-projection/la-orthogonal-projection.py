import numpy as np

def projection_matrix(A):
    """
    Returns: ndarray, the projection matrix onto the column space of A.
    """
    A = np.array(A, dtype = np.float64)
    A_pinv = np.linalg.pinv(A)
    P = A@A_pinv
    return P