import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    data = np.array(data, dtype = np.float64)
    return np.stack([np.sort(data,axis = axis), np.argsort(data, axis = axis)], axis = 0)