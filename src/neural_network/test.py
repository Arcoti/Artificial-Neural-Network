import numpy as np

from .solve import solve

# Y here is simply the labels
def test(X, Y, parameters):
    total_correct = 0
    total_predictions = 0

    for batch_X, batch_Y in zip(X, Y):
        # Solve
        results = solve(batch_X, parameters)

        # Calculate Number of Correct Predictions
        results_int = results.astype(int)
        label_int = batch_Y.astype(int)
        total_correct += np.sum((results_int == label_int).astype(int))

        # Calculate Number of test cases
        total_predictions += results.shape[0]

    # Calculate Accuracy Score
    accuracy_score = total_correct / total_predictions

    return accuracy_score