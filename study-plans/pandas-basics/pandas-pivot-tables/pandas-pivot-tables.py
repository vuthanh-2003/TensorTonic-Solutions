import pandas as pd

def create_pivot(data, index, columns, values, aggfunc):
    """
    Returns: nested dict {column_value: {index_value: agg_result}}
    """
    df = pd.DataFrame(data)
    pt = pd.pivot_table(df, values = values, index = index, columns = columns, aggfunc = aggfunc, fill_value = 0)
    result = pt.to_dict()
    return result
    