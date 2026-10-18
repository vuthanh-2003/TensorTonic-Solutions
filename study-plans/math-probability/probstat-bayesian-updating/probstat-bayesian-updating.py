def bayesian_update(prior_alpha: float, prior_beta: float, successes: int, failures: int) -> dict:
    """
    Returns posterior parameters and prior and posterior means.
    """
    prior_mean = prior_alpha/(prior_alpha+prior_beta)
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + failures
    post_mean = post_alpha/(post_alpha+post_beta)
    return {"posterior_alpha": post_alpha, "posterior_beta": post_beta, "posterior_mean": post_mean, "prior_mean": prior_mean}
