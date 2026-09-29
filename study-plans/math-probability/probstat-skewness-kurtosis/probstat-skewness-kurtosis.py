import numpy as np

def skewness_kurtosis(data: list) -> dict:
    """
    Returns adjusted statistics and their interpretations in a dictionary.
    """
    values = np.asarray(data, dtype=np.float64)
    n = values.size
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    skewness = n / ((n - 1) * (n - 2)) * np.sum(((values - mean) / std) ** 3)
    kurtosis = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((values - mean) / std) ** 4)
    kurtosis -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    skewness = round(float(skewness), 4)
    kurtosis = round(float(kurtosis), 4)
    skew_label = "right-skewed" if skewness > 0.5 else "left-skewed" if skewness < -0.5 else "approximately symmetric"
    kurtosis_label = "leptokurtic" if kurtosis > 1 else "platykurtic" if kurtosis < -1 else "mesokurtic"
    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "skew_interpretation": skew_label,
        "kurtosis_interpretation": kurtosis_label,
    }