import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """
    result = {}
    df = pd.DataFrame(data)
    shape = list(df.shape)
    columns = df.columns.to_list()
    result['data'] = data
    result['shape'] = shape
    result['columns'] = columns
    return result