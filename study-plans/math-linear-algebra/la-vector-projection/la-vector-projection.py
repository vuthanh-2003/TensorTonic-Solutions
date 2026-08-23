import numpy as np

def vector_projection(u, v):
    """
    Returns: float64 array, the projection of u onto v.
    """
    u = np.array(u)
    v = np.array(v)
    proj = (np.dot(u,v)/np.dot(v,v))*v
    return proj