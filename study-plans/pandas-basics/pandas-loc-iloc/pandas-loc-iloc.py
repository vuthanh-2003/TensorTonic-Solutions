import pandas as pd

def iloc_selection(data, row, col):
    """
    Returns: list [element, row_values, col_values]
    """
    result = []
    df = pd.DataFrame(data)
    result.append(df.iloc[row, col])
    result.append(df.iloc[row,:])
    result.append(df.iloc[:, col])
    return result