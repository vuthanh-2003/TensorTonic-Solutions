import pandas as pd

def unstack_to_wide(data, index_col, columns_col, values_col):
    """
    Returns: dict with index_col plus one key per unique value in columns_col
    """
    df = pd.DataFrame(data)
    df_group = df.groupby([index_col, columns_col])[values_col].agg(lambda x: x)
    result = df_group.unstack(fill_value = 0).reset_index().to_dict('list')
    return result