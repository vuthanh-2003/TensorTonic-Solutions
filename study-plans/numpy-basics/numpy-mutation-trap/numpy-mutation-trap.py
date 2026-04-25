import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    data = np.array(data, dtype = np.float64)
    arr = data[row_idx, :].copy()
    arr_1 = np.clip(arr, lo, hi)
    result = np.vstack([arr, arr_1])
    return result