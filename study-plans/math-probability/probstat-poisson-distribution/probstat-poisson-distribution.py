from math import exp, factorial

def poisson_distribution(lam: float, max_k: int) -> list:
    """
    Returns the PMF list, cumulative probability, and zero-event probability.
    """
    results = []
    pmf = []
    cum_prob = 0
    for i in range(max_k+1):
        p_i = (((lam)**i)*exp(-1*lam))/factorial(i)
        pmf.append(p_i)
        cum_prob += p_i
    results.append(pmf)
    results.append(cum_prob)
    results.append(pmf[0])
    return results
