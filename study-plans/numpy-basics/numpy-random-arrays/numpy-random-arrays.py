import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    np.random.seed(seed) 
    
    if kind == 'uniform':
        return np.random.uniform(low=0.0, high=1.0, size=shape)
    else:
        return np.random.normal(loc=0.0, scale=1.0, size=shape)
