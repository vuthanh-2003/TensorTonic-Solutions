import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    data = np.array(data, dtype = np.float64)
    data_1 = np.where(data > threshold, np.ones(data.shape), 0)
    data_2 = data.copy()
    for i in range(data_2.shape[0]):
        if np.any(data_2[i] > threshold) == False:
            data_2[i] = 0
        else:
            continue
    data_3 = data.copy()
    for i in range(data_3.shape[0]):
        if np.all(data_3[i] > threshold) == False:
            data_3[i] = 0
        else:
            continue
    return np.stack([data_1, data_2, data_3], axis=0)