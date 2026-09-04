import numpy as np

def pseudoinverse(A):
    """
    Returns: ndarray, the Moore-Penrose pseudoinverse of A.
    """
    A = np.array(A)
    A_pinv = np.linalg.pinv(A)
    return A_pinv