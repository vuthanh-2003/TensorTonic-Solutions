import pandas as pd

def reset_index_demo(data, index_col):
    """
    Returns: list [columns_before_reset, columns_after_reset]
    """
    result = []
    df = pd.DataFrame(data).set_index(index_col)
    result.append(df.columns.tolist())
    df = df.reset_index().rename(columns={'index': index_col})
    result.append(df.columns.tolist())
    return result