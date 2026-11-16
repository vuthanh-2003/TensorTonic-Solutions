import numpy as np

def hessian_convexity_analysis(X: list, y: list) -> dict:
    """
    Returns the half-MSE Hessian, eigenvalues, and convexity.
    """
    X = np.array(X,dtype=np.float64)
    n,d = X.shape
    hessian = (X.T@X)/n
    eigenvalues = np.linalg.eigvalsh(hessian)
    h = -1e-10
    is_convex = np.all(eigenvalues>=h)
    return {"hessian": hessian.tolist(), "eigenvalues": eigenvalues.tolist(), "is_convex": is_convex}