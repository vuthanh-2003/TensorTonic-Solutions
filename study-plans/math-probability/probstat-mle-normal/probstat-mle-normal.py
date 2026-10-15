import numpy as np

def mle_normal(data: list) -> dict:
    """
    Returns the normal-distribution MLEs mu_mle and sigma_mle.
    """
    data = np.array(data)
    mu_mle = np.sum(data)/len(data)
    sigma_mle = np.sqrt((1/len(data))*np.dot(data-mu_mle,data-mu_mle))
    return {"mu_mle": mu_mle, "sigma_mle": sigma_mle}
