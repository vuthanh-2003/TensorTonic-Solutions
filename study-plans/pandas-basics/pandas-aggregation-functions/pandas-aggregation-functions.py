import pandas as pd

def multi_agg(data, group_col, value_col, funcs):
    """
    Returns: dict mapping function name to {group: value} dict
    """
    df = pd.DataFrame(data)
    df_new = df.groupby(group_col).agg(**{i : (value_col, i) for i in funcs}).to_dict()
    return df_new