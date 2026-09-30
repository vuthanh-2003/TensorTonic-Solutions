import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the float64 feature-correlation matrix.
    """
    X = np.asarray(X, dtype = np.float64)
    m,n = X.shape
    corr = np.zeros((n,n))
    for i in range(n):
        X_i = X[:,i].copy()
        mean_i = X[:,i].mean()
        corr[i,i] = (np.dot(X_i - mean_i, X_i - mean_i))/(np.sqrt(np.dot(np.sum((X_i-mean_i)**2), np.sum((X_i - mean_i)**2))))
        for j in range(i+1,n):
            X_j = X[:,j].copy()
            mean_j = X[:,j].mean()
            corr[i,j] = (np.dot(X_i - mean_i, X_j - mean_j))/(np.sqrt(np.dot(np.sum((X_i-mean_i)**2), np.sum((X_j - mean_j)**2))))
            corr[j,i] = corr[i,j]
    return corr