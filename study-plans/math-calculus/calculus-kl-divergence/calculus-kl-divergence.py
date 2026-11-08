import numpy as np

def kl_divergence_analysis(P: list, Q: list, mu1: float, sigma1: float, mu2: float, sigma2: float, n: int) -> dict:
    """
    Returns discrete, numerical Gaussian, and analytical Gaussian KL values.
    """
    P = np.array(P, dtype = np.float64)
    Q = np.array(Q, dtype = np.float64)
    valid = P >0
    discrete_kl = np.sum(P[valid]*np.log(P[valid]/Q[valid]))
    analytical_kl = np.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 1/2
    a = min(mu1,mu2) - 5*max(sigma1,sigma2)
    b = max(mu1,mu2) + 5*max(sigma1,sigma2)
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    p_x = (1/(sigma1 * np.sqrt(2*np.pi)))*np.exp(-(x-mu1)**2 / (2*sigma1**2))
    log_ratio = np.log(sigma2 / sigma1) - (x - mu1)**2 / (2 * sigma1**2) + (x - mu2)**2 / (2 * sigma2**2)
    f_x = p_x*log_ratio
    weights = np.ones(n+1)
    weights[1:n:2] = 4
    weights[2:n:2] = 2
    numerical_kl = (h/3)*np.sum(weights*f_x)
    return {
        "discrete_kl": float(discrete_kl),
        "numerical_kl": float(numerical_kl),
        "analytical_kl": float(analytical_kl)
    }
    