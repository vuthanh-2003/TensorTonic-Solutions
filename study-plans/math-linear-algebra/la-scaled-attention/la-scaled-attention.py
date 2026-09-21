import numpy as np

def scaled_dot_product_attention(Q: list, K: list, V: list) -> np.ndarray:
    """
    Returns the scaled-attention output as a float64 array.
    """
    Q = np.array(Q,dtype=np.float64)
    K = np.array(K, dtype = np.float64)
    V = np.array(V, dtype = np.float64)
    n,d_k = K.shape
    S = (Q@K.T)/np.sqrt(d_k)
    scaled = np.exp(S - np.max(S, axis = 1, keepdims = True))
    sum_scaled = np.sum(scaled, axis = 1, keepdims = True)
    P = scaled/sum_scaled
    O = P@V
    return O