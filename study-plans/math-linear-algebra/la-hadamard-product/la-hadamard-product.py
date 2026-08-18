import numpy as np

def hadamard_product(A, B):
    """
    Returns: ndarray, the element-wise product A * B.
    """
    A = np.array(A)
    B = np.array(B)
    C = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i,j] = A[i,j] * B[i,j]
    return C
    