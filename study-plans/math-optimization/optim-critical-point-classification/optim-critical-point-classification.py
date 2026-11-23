import numpy as np

def classify_critical_point(H: list) -> str:
    """
    Returns the second-order classification string.
    """
    H = np.array(H, dtype = np.float64)
    eig_values = np.linalg.eigvalsh(H)
    if np.any(np.abs(eig_values)<=1e-6):
        return "degenerate"
    elif np.all(eig_values>0):
        return "local_min"
    elif np.all(eig_values<0):
        return "local_max"
    else:
        return "saddle"