import numpy as np

def convex_set_membership(A: list, b: list, x: list) -> dict:
    """
    Returns the membership decision and maximum violation in a dictionary.
    """
    A = np.array(A, dtype = np.float64)
    b = np.array(b, dtype = np.float64)
    x = np.array(x, dtype = np.float64)

    r = A@x - b
    v = round(float(np.max(r)),6)
    in_set = bool(v <= 1e-6)
    return {"in_set": in_set, "max_violation": v}
    