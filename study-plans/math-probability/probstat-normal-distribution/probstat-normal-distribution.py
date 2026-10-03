from scipy.stats import norm

def normal_distribution(mu: float, sigma: float, x: float) -> dict:
    """
    Returns the z-score, CDF, PDF, and one-standard-deviation probability.
    """
    pdf = norm.pdf(x,loc = mu, scale = sigma)
    cdf = norm.cdf(x,loc = mu, scale = sigma)
    cdf_high_1_sigma = norm.cdf(mu+sigma,loc = mu, scale = sigma)
    cdf_low_1_sigma = norm.cdf(mu-sigma,loc = mu, scale = sigma)
    prob_within_1_std = cdf_high_1_sigma - cdf_low_1_sigma
    z_score = (x-mu)/sigma
    return {"cdf": cdf, "pdf": pdf, "prob_within_1_std":prob_within_1_std, "z_score":z_score}