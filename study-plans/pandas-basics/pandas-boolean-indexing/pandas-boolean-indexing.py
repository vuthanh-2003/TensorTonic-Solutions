import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    result = {}
    df = pd.DataFrame(data)
    df = df[df[column] > threshold]
    count = df.shape[0]
    filtered_data = df.to_dict('list')
    result['filtered_data'] = filtered_data
    result['count'] = count
    return result