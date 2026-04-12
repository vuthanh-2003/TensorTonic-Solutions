import pandas as pd

def rename_columns(data, rename_map):
    """
    Returns: dict mapping renamed column names to value lists
    """
    df = pd.DataFrame(data)
    df = df.rename(columns = rename_map)
    result = df.to_dict('list')
    return result