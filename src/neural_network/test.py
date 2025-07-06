import numpy as np

from .solve import solve

# Y here is simply the labels
def test(X, Y, parameters):
    # Solve
    results = solve(X, parameters)

    # Calculate Number of Correct Predictions
    results_int = results.astype(int)
    label_int = Y.astype(int)
    total_correct = np.sum((results_int == label_int).astype(int))

    # Calculate Accuracy Score
    total_predictions = results.shape[0]
    accuracy_score = total_correct / total_predictions

    return accuracy_score