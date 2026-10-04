import numpy as np

def standard_errors(samples: list) -> dict:
    """
    Returns each sample standard error and their mean.
    """
    se_list = []
    for sample in samples:
        sample = np.array(sample, dtype = np.float64)
        std = np.std(sample,ddof = 1)
        se_list.append(std/np.sqrt(len(sample)))
    mean_se = np.mean(np.array(se_list))
    return {"mean_se": mean_se, "standard_errors": se_list}
