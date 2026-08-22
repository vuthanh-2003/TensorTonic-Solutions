import numpy as np

def matrix_rank(A):
    """
    Returns: int, the rank of matrix A.
    """
    A = np.array(A)
    singular_values = np.linalg.svd(A, compute_uv = False)
    return int(np.sum(singular_values>1e-10))