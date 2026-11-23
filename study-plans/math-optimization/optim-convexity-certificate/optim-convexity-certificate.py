import numpy as np

def convexity_certificate(H: list) -> dict:
    """
    Returns the PSD decision and smallest eigenvalue in a dictionary.
    """
    H = np.array(H, dtype = np.float64)
    eigen_value = np.linalg.eigvalsh(H)
    eig_min = round(float(np.min(eigen_value)),6)
    is_convex = bool(eig_min >= -1e-6)
    return {"is_convex": is_convex, "min_eigenvalue": eig_min}
    