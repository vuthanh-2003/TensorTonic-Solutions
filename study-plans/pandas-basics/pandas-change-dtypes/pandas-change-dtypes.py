import pandas as pd

def change_dtype(data, column, target_type):
    """
    Returns: list [dtypes_before, dtypes_after] (both dicts)
    """
    result = []
    df = pd.DataFrame(data)
    result.append(df.dtypes.astype('str').to_dict())
    df = df.astype({column:target_type})
    result.append(df.dtypes.astype('str').to_dict())
    return result