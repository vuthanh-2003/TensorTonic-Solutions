def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    result = []
    counts_dict = {}
    n = len(values)
    for i in values:
        if i not in counts_dict:
            counts_dict[i] = 1
        else:
            counts_dict[i] += 1
    for i in values:
        if i in counts_dict:
            result.append(counts_dict[i]/n)
    return result