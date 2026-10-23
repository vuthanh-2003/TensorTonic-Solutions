import numpy as np

def sigmoid_squeeze_analysis(x: list) -> dict:
    """
    Returns sigmoid bounds and saturation flags in a dictionary.
    """
    x = np.array(x, dtype = np.float64)
    sigmoid = 1/(1+np.exp(-1*x))
    lower_bound = np.maximum(0, 1-np.exp(-1*x))
    upper_bound = np.minimum(1,np.exp(x))
    bounds = np.column_stack((lower_bound,sigmoid,upper_bound))
    is_saturated = np.minimum(sigmoid, 1 - sigmoid) < 0.0001
    return {"bounds": bounds.tolist(), "is_saturated": is_saturated.tolist()}
        
    