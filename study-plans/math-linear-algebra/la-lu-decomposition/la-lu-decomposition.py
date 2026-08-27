import numpy as np
import scipy.linalg as la

def lu_decomposition(A):
    """
    Returns: tuple (L, U) where A = L @ U.
    """
    A = np.array(A, dtype = np.float64)
    n =  A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for j in range(n):
        for i in range(j+1, n):
            m = U[i,j]/U[j,j]
            L[i,j] = m
            U[i,j:] -= m*U[j,j:]
    return L,U