def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = recommended[:k]
    relevant_set = set(relevant)
    counts = sum(1 for i in top_k if i in relevant_set)
    precision = counts/k
    recall = counts/len(relevant)
    return [precision, recall]