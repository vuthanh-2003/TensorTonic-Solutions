from math import comb

def binomial_distribution(n: int, p: float, threshold: int) -> dict:
    """
    Returns the PMF, mean, variance, and probability at least threshold.
    """
    pmf = []
    prob_at_least = 0
    mean = n*p
    variance = n*p*(1-p)
    for i in range(n+1):
        pmf.append(comb(n,i)*(p**i)*((1-p)**(n-i)))
        if i >= threshold:
            prob_at_least += pmf[-1]
    return {"mean": mean, "pmf": pmf, "prob_at_least": prob_at_least,"variance": variance}
