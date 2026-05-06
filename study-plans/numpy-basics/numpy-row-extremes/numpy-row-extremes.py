import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    data = np.array(data, dtype = np.float64)
    result = np.row_stack([data.max(axis = 1), data.argmax(axis = 1), data.min(axis = 1), data.argmin(axis = 1)])
    return result