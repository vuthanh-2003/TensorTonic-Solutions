import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    arr = np.array(arr, dtype = np.float64)
    result = np.take(arr, indices, axis = axis)
    return result