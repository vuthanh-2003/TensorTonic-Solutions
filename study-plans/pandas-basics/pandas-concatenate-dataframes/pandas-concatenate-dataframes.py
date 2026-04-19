import pandas as pd

def concat_dataframes(dfs):
    """
    Returns: list [shape, data] where shape is [rows, cols]
    """
    data = [pd.DataFrame(df) for df in dfs]
    result = pd.concat(data)
    return [[result.shape[0], result.shape[1]], result.to_dict('list')]