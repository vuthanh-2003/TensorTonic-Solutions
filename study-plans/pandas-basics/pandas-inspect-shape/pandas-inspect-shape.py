import pandas as pd

def inspect_dataframe(data):
    """
    Returns: dict with 'rows', 'cols' (ints), 'columns' (list),
    'dtypes' (dict), 'total_values' (int)
    """
    result = {}
    df = pd.DataFrame(data)
    rows=df.shape[0]
    cols = df.shape[1]
    columns = df.columns.tolist()
    dtypes = dtypes = {i:str(j) for i,j in zip(df.columns,df.dtypes)}
    total_values = df.size
    result['rows'] = rows
    result['cols'] = cols
    result['columns'] = columns
    result['dtypes'] = dtypes
    result['total_values'] = total_values 
    return result