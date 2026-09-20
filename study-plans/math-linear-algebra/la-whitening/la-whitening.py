import numpy as np

def whiten(X: list) -> np.ndarray:
    """
    Returns the whitened samples as a float64 array.
    """
    X = np.array(X, dtype = np.float64)
    X_mean = X.mean(axis = 0)
    X_centered  = X - X_mean
    n = X.shape[0]
    cov = (1/(n-1))*X_centered.T@X_centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    for i in range(eigenvectors.shape[1]):
        max_idx = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[max_idx, i] < 0:
            eigenvectors[:, i] *= -1
    scales = np.array([1.0 / np.sqrt(value) if value > 1e-10 else 0.0 for value in eigenvalues], dtype=np.float64)
    return np.asarray((X_centered @ eigenvectors) * scales, dtype=np.float64)
    