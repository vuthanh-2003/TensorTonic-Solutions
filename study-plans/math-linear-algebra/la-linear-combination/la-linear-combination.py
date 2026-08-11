import numpy as np

def linear_combination(vectors, coefficients):
    """
    Returns: float64 array, the weighted sum of vectors.
    """
    vectors = np.array(vectors)
    coefficients = np.array(coefficients)
    combine = coefficients@vectors
    return combine