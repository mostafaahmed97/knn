import numpy as np

def distance_metric(p):
    """
    Returns the function to compute distance between 
    sample and all members of X

    Uses the formula  [Σ (|X - Y|)^p] ^ (1/p)

    Different values of p produce different metrics
    p = 1, Manhattan distance
    p = 2, Euclidean distance 
    and so on
    """
    def calculate_distance(X, sample):
        difference = np.abs(X - sample) ** p
        distance = np.sum(difference, axis=1) ** (1/p)
        return distance

    return calculate_distance

def knn(data, labels, test_data, k=3, p=2):
    """
    Predicts label for test_data based on nearest K neighbours

    Parameters
    ----------
    data: array_like
        Data matrix, of size N Samples * M Features

    labels: array
        Labels for samples, of size N * 1

    test_data: array_like
        Data to be used for predictions

    k : int
        Number of nearby neighbours to decide label
    
    p: int
        Used to generate the distance metric function
    """

    calc_dist = distance_metric(p)
    predictions = []

    for observation in test_data:
        distances = calc_dist(data, observation)
        min_idxs = np.argpartition(distances, k)[:k]

        nearby_labels = labels[min_idxs]
        dominant_label = np.argmax(np.bincount(nearby_labels))

        predictions.append(dominant_label)

    return np.array(predictions)
