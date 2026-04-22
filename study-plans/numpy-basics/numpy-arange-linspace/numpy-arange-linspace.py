import numpy as np

def create_sequence(start, stop, param, kind):
    """
    Returns: 1D ndarray of float64 values
    """
    if kind == 'arange':
        return np.arange(start = start, stop = stop, step = param, dtype = np.float64)
    else:
        return np.linspace(start = start, stop = stop, num = param)
