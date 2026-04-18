import pandas as pd

def merge_dataframes(left, right, on, how):
    """
    Returns: dict of column to value lists
    """
    df_left, df_right = pd.DataFrame(left), pd.DataFrame(right)
    result = pd.merge(df_left, df_right, on = on, how = how).to_dict("list")
    return result