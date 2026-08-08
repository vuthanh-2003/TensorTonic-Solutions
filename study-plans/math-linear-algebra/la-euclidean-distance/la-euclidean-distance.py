import numpy as np

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.array(x, dtype = np.float32)
    y = np.array(y, dtype = np.float32)
    return np.sqrt(np.sum((x-y)**2))