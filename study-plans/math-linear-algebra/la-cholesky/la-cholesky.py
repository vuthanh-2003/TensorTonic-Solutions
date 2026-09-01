import numpy as np

def cholesky_decompose(A):
    """
    Returns: lower triangular L where A = L @ L.T, or None if not positive definite.
    """
    A = np.array(A, dtype = np.float64)
    if np.allclose(A,A.T) == False:
        return None
    n = A.shape[0]
    for a in range(1,n+1):
        sub_matrix = A[:a, :a]
        if np.linalg.det(sub_matrix) <= 0:
            return None
    L = np.zeros((n,n))
    for j in range(n):
        s = 0.0
        for k in range(j):
            s += L[j,k]**2
        d = A[j,j] - s
        if d <0:
            return None
        L[j,j] = np.sqrt(d)
        for i in range(j+1,n):
            s = 0.0
            for k in range(j):
                s += L[i,k]*L[j,k]
            L[i,j] = (A[i,j]-s)/L[j,j]
    return L