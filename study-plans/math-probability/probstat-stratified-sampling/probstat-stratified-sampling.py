from math import floor, sqrt

def stratified_sample(strata_means: list, strata_stds: list, strata_sizes: list, total_sample: int) -> dict:
    """
    Returns exact allocations, the stratified mean, and its standard error.
    """
    total_population = sum(strata_sizes)
    weights = [size / total_population for size in strata_sizes]
    ideal = [total_sample*w for w in weights]
    allocations = [max(1,floor(a)) for a in ideal]
    while sum(allocations) > total_sample:
        candidates = [i for i,n in enumerate(allocations) if n>1]
        i = min(candidates, key = lambda j: ideal[j] - floor(ideal[j]))
        allocations[i] -= 1
    remainders = [ideal[i] - floor(ideal[i]) for i in range(len(ideal))]
    while sum(allocations) < total_sample:
        candidates = sorted(range(len(allocations)), key = lambda i: remainders[i], reverse = True)
        for i in candidates:
            if sum(allocations) == total_sample:
                break
            allocations[i] += 1
            remainders[i] = -1
    stratified_mean = sum(w*mean for w,mean in zip(weights, strata_means))
    variance = sum((w**2)*(std**2)/n for w,std,n in zip(weights, strata_stds, allocations))
    standard_err = sqrt(variance)
    return {
        "allocations": allocations,
        "stratified_mean": stratified_mean,
        "stratified_se": standard_err
    }
