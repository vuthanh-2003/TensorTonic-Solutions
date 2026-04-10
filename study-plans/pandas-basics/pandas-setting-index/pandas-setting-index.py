import pandas as pd

def set_index_column(data, index_col):
    """
    Returns: dict with 'index_values', 'columns', 'data'
    """
    result = {}
    df = pd.DataFrame(data)
    index_values = list(df[index_col])
    df = df.set_index(index_col)
    columns = df.columns.tolist()
    data_new = df.to_dict('list')
    result['index_values'] = index_values
    result['columns'] = columns
    result['data'] = data_new
    return result
    