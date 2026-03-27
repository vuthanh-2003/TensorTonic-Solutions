import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    small_entropy = []
    unique_val, counts = np.unique(y, return_counts = True)
    counts = np.array(counts)
    sum_counts = np.sum(counts)
    for i in range(0, len(counts)):
        if counts[i] == 0:
            small_entropy.append(0)
        else:
            small_entropy.append((counts[i]/sum_counts)*np.log2(counts[i]/sum_counts))
    small_entropy = np.array(small_entropy)
    entropy = - np.sum(small_entropy)
    return entropy
        