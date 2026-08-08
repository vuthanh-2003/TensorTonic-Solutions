import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.array(a,dtype=np.float32)
    b = np.array(b,dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return (a@b)/(np.linalg.norm(a)*np.linalg.norm(b))