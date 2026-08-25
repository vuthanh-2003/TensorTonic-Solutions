import numpy as np

def gram_schmidt(vectors):
    """
    Returns: float64 array of shape (k, n), orthonormal basis spanning the input space.
    """
    vectors = np.array(vectors)
    n = vectors.shape[0]
    Q = np.zeros_like(vectors, dtype = np.float64)
    for k in range(n):
        u = vectors[k].astype(float)
        for j in range(k):
            u -= np.dot(Q[j],u)*Q[j]
        Q[k] = u/np.linalg.norm(u)
    return Q