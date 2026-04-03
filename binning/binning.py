def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    min_val = min(values)
    max_val = max(values)
    bin_val = []
    if max_val == min_val:
        for i in range(len(values)):
            bin_val.append(0)
    else:
        bin_width = (max_val - min_val)/num_bins
        for v in values:
            bin_idx = int((v-min_val)/bin_width)
            bin_val.append(min(bin_idx, num_bins - 1))
    return bin_val
        