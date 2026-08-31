import numpy as np

def eigendecompose(A):
    """
    Returns: tuple (eigenvalues, eigenvectors), sorted by descending magnitude.
    """
    A = np.array(A, dtype = np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    indices = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:,indices]
    eigenvectors = eigenvectors/np.linalg.norm(eigenvectors, axis = 0, keepdims = True)
    return eigenvalues,eigenvectors