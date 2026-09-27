import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns mean, median, and mode as Python floats in a dictionary.
    """
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    mean = float(np.sum(x)/n)
    if n%2 != 0:
        median = float(x[n//2])
    else:
        median = float((x[n//2 - 1] + x[n//2])/2)
    values,counts = np.unique(x, return_counts = True)
    mode = float(values[np.argmax(counts)])
    return {"mean": mean, "median": median, "mode": mode}
    