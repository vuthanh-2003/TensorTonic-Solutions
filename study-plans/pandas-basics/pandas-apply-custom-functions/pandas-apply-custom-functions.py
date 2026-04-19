import pandas as pd

def apply_transform(data, column, operation):
    """
    Returns: dict with original columns plus column_transformed
    """
    df = pd.DataFrame(data)
    if operation == 'normalize':
        df[f'{column}_transformed'] = (
    (df[column] - df[column].min()) /
    (df[column].max() - df[column].min())
).round(4)
    elif operation == 'double':
        df[f'{column}_transformed'] = df[column]*2
    else:
        df[f'{column}_transformed'] = df[column].apply(operation)
    df = df.to_dict('list')
    return df
    