import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)
    sum_prob =  np.sum(p, axis = 0)
    if np.allclose(sum_prob, 1, atol = 1e-06):
        expected_value = x@p.T
        return expected_value
    else:
        raise ValueError
