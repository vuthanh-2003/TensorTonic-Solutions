import numpy as np

def softmax_stability_analysis(z: list) -> dict:
    """
    Returns direct and stable softmax results with an issue flag.
    """
    logits = np.asarray(z, dtype=np.float64)
    exponentials = np.exp(logits)
    naive = exponentials / np.sum(exponentials)
    shifted = np.exp(logits - np.max(logits))
    stable = shifted / np.sum(shifted)
    return {
        "naive": naive.tolist(),
        "stable": stable.tolist(),
        "naive_has_issues": bool(np.any(~np.isfinite(naive))),
    }