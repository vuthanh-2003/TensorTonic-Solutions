import numpy as np

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    predictions = []

    for x in X_test:
        # 1. Tính bình phương khoảng cách Euclidean
        distances = np.sum((X_train - x) ** 2, axis=1)
        indices = np.lexsort((np.arange(len(X_train)), distances))
        neighbors = y_train[indices[:k]]
        labels, counts = np.unique(neighbors, return_counts=True)
        prediction = labels[np.argmax(counts)]
        predictions.append(int(prediction))
    return predictions

