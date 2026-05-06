import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    data = np.array(data, dtype = np.float64)
    lo = np.percentile(data, lo_q, axis = 0)
    hi = np.percentile(data, hi_q, axis = 0)
    arr = np.clip(data, lo, hi)
    lo_mask = np.where(data<lo, 1, 0)
    hi_mask = np.where(data>hi,1,0)
    return np.stack([arr, lo_mask, hi_mask], axis = 0)