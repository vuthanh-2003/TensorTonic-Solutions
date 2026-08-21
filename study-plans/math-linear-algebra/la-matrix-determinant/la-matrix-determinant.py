import numpy as np

def matrix_determinant(A):
    """
    Returns: float, the determinant of square matrix A.
    """
    A = np.array(A)
    if A.shape[0] == 1:
        return A[0,0]
    if A.shape[0] == 2:
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    det = 0.0
    for j in range(A.shape[1]):
        sign = (-1)**j
        minor = np.delete(A,0,axis = 0)
        minor = np.delete(minor,j, axis = 1)
        det += sign*A[0,j]*matrix_determinant(minor)
    return det