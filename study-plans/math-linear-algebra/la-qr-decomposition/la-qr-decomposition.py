import numpy as np

def qr_decompose(A):
    """
    Returns: tuple (Q, R) where A = Q @ R.
    """
    A = np.array(A, dtype = np.float64)
    m,n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    for j in range(min(m,n)):
        v = A[:,j].copy()
        for i in range(j):
            R[i,j] = np.dot(Q[:,i],A[:,j])
            v -= R[i,j]*Q[:,i]
        R[j,j] = np.linalg.norm(v)
        if R[j,j] != 0:
            Q[:,j] = v/R[j,j]
        if R[j,j] < 0:
            R[j,j] = -R[j,j]
            R[j,j:] = - R[j,j:]
            Q[:,j] = -Q[:,j]
    return Q,R