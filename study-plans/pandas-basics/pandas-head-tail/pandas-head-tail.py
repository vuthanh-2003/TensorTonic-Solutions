import pandas as pd

def head_tail(data, n):
    """
    Returns: dict with 'head' and 'tail' (both dicts of column -> list)
    """
    result = {}
    head = {}
    tail = {}
    df = pd.DataFrame(data)
    for i in df.columns.tolist():
        head[i] = list(df[i].head(n))
        tail[i] = list(df[i].tail(n))
    result['head'] = head
    result['tail'] = tail
    return result