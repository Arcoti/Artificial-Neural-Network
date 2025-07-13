import numpy as np
from .solve import solve
from .persist import load_model

def test(X: list, Y: list, model_path: str):
    # Initialize Variables
    total_correct = 0
    total_predictions = 0

    # Load the parameters
    parameters = load_model(model_path)

    # Test in batches
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
