import numpy as np

def matrix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    A = np.array(A)
    diag = 0
    total_diag = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                diag = A[i,j]
                total_diag+=diag
    return total_diag