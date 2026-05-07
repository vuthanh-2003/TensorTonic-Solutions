import numpy as np

def quantize_and_frame(data, decimals, pad_width):
    """Returns: np.ndarray of shape (3, m+2p, n+2p), stacked rounded, floored, ceiled with zero-padding"""
    data = np.array(data, dtype = np.float64)
    result = []
    for func in [np.round, np.floor, np.ceil]:
        if func == np.round:
            quantized = func(data, decimals)
        else:
            quantized = func(data)
        padded = np.pad(quantized, pad_width, mode = 'constant', constant_values = 0)
        result.append(padded)
    output = np.stack(result)
    return output