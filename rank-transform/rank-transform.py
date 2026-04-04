def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    # Write code here
    n = len(values)
    result = [i for i in range(n)]
    indices = [i for i in range(n)]
    indices.sort(key = lambda i:values[i])
    sorted_indice = indices
    i = 0
    while i < n:
        j = i
        while j < n and values[sorted_indice[j]] == values[sorted_indice[i]]:
            j += 1
        start_rank = i + 1
        end_rank = j
        avg_rank = (start_rank + end_rank)/2
        for k in range(i, j):
            original_index = sorted_indice[k]
            result[original_index] = avg_rank
        i = j
    return result
            
            
        
        