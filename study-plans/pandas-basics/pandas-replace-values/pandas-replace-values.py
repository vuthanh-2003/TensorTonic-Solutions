import pandas as pd

def replace_values(data, column, old_val, new_val):
    """
    Returns: dict with 'data' (dict) and 'count' (int)
    """
    result = {}
    df = pd.DataFrame(data)
    count = len(df[df[column] == old_val])
    df[column] = df[column].replace(old_val, new_val)
    result['data'] = df.to_dict('list')
    result['count'] = count
    return result