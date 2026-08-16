import numpy as np

def matrix_transpose(A):
    """
    Returns: ndarray, the transpose of A.
    """
    A = np.array(A)
    A_t = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A_t.shape[0]):
        for j in range(A_t.shape[1]):
            A_t[i,j] = A[j,i]
    return A_t
        