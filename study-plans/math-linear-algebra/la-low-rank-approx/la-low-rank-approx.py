import numpy as np

def low_rank_approximation(A, r):
    """
    Returns: float64 ndarray of shape (m, n), the best rank-r approximation of A.
    """
    U, s, Vt = np.linalg.svd(A,full_matrices = False)
    A_r = U[:,:r]@np.diag(s[:r])@Vt[:r,:]
    return A_r