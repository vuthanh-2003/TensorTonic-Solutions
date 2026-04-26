import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    arr = np.array(data, dtype = np.float64)
    col_mean = arr.mean(axis = 0)
    col_std = arr.std(axis = 0)
    standardized = (arr - col_mean)/col_std
    return standardized