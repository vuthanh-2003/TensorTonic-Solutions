import pandas as pd

def select_columns(data, columns):
    """
    Returns: dict mapping selected column names to value lists
    """
    df = pd.DataFrame(data)
    result = df[columns].to_dict('list')
    return result