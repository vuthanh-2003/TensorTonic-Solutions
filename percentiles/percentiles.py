import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.array(x)
    q = np.array(q)
    percentiles = np.percentile(x,q, method = 'linear')
    return percentiles