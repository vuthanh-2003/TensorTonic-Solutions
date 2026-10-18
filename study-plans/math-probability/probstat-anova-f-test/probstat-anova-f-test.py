import numpy as np
from scipy.stats import f

def anova_f_test(groups: list) -> list:
    """
    Returns the F-statistic, degrees of freedom, p-value, and decision.
    """
    groups = np.array(groups, dtype=object)  # Cho phép các nhóm có thể khác kích thước
    k = len(groups)

  # Tính tổng số quan sát N và trung bình chung (grand mean)
    all_data = np.concatenate([np.array(g) for g in groups])
    N = len(all_data)
    grand_mean = all_data.mean()
    ss_between = 0
    ss_within = 0
    for g in groups:
        g_arr = np.array(g)
        n_j = len(g_arr)
        g_mean = g_arr.mean()

    # SS between: n_j * (trung bình nhóm - trung bình chung)^2
        ss_between += n_j * (g_mean - grand_mean) ** 2

    # SS within: tổng bình phương độ lệch của từng phần tử so với trung bình nhóm của nó
        ss_within += np.sum((g_arr - g_mean) ** 2)
    df_between = k - 1
    df_within = N - k
    f_stats = (ss_between / df_between) / (ss_within / df_within)
    p_value = f.sf(f_stats, df_between, df_within)
    sig = p_value < 0.05
    return [f_stats, df_between, df_within, p_value, sig]
