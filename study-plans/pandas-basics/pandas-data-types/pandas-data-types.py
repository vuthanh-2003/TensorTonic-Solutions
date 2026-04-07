import pandas as pd

def data_types_overview(data):
    """
    Returns: dict with 'dtypes', 'type_counts', 'num_columns'
    """
    result = {}
    df = pd.DataFrame(data)
    dtypes = {i:str(j) for i,j in zip(df.columns, df.dtypes)}
    type_counts = {}
    for i in dtypes.values():
        if i not in type_counts:
            type_counts[i] = 1
        else:
            type_counts[i] += 1
    num_columns = len(df.columns.tolist())
    result['dtypes'] = dtypes
    result['type_counts'] = type_counts
    result['num_columns'] = num_columns
    return result