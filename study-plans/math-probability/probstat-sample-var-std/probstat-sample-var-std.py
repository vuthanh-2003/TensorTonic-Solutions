import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns sample variance and standard deviation as Python floats.
    """
    x = np.array(x)
    mean = x.mean()
    n = len(x)
    s_var = float((1/(n-1))*np.dot((x-mean),(x-mean)))
    s_std = float(np.sqrt(s_var))
    return {"variance": s_var, "std_dev": s_std}