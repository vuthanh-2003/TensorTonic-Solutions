def perms_and_combs(n: int, r: int) -> list:
    """
    Returns permutations, combinations, and n factorial as integers.
    """
    results = []
    def factorial(x):
        if x == 1 or x == 0:
            return 1
        else:
            return x*factorial(x-1)
    n_factorial = factorial(n)
    r_factorial = factorial(r)
    n_r_factorial = factorial(n-r)
    results.append(n_factorial//n_r_factorial)
    results.append(n_factorial//(r_factorial*n_r_factorial))
    results.append(n_factorial)
    return results