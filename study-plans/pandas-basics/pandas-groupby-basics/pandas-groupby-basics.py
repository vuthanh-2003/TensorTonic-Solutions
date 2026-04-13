import pandas as pd

def groupby_basics(data, group_col, value_col):
    """
    Returns: dict with 'sum', 'mean', 'count' (each a dict)
    """
    df = pd.DataFrame(data)
    df_new = df.groupby(group_col)[[value_col]].agg(['sum', 'mean', 'count'])
    result = {
        'sum': df_new.iloc[:, 0].to_dict(),
        'mean': df_new.iloc[:, 1].to_dict(),
        'count': df_new.iloc[:, 2].to_dict()
    }
    return result