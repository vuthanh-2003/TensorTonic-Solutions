import numpy as np

def least_squares(A, b):
    """
    Returns: float64 array, the solution minimizing ||A @ x - b||^2.
    """
    A = np.array(A)
    b = np.array(b)
    tol = 1e-10
    U, s, Vt = np.linalg.svd(A)
    s_inv = np.array([1/x if x > tol else 0 for x in s])
    Sigma_plus = np.zeros((A.shape[1],A.shape[0]))
    np.fill_diagonal(Sigma_plus,s_inv)
    A_pinv = Vt.T@Sigma_plus@U.T
    return A_pinv@b