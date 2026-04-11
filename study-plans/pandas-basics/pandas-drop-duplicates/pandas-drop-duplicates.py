import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)
    result = []
    result.append(df.shape[0])
    df = df.drop_duplicates(keep = 'first')
    df_dict = df.to_dict('list')
    result.append(df.shape[0])
    result.append(df_dict)
    return result