from math import exp

def exponential_distribution(lam: float, t: float) -> dict:
    """
    Returns the PDF, CDF, survival probability, mean, and variance.
    """
    pdf = lam*exp(-1*lam*t)
    cdf = 1 - exp(-1*lam*t)
    survive_p = 1 - cdf
    mean = 1/lam
    variance = 1/(lam**2)
    return {"cdf": cdf, "mean":mean, "pdf": pdf, "survival":survive_p, "variance": variance}
