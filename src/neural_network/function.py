import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(np.dot(-1, Z)))

def sigmoid_first_derivative(Z: np.ndarray):
    return sigmoid(Z) * (1-sigmoid(Z))

def cost_function(A: np.ndarray, Y: np.ndarray):
    m = Y.shape[1]

    # Binary Cross Entropy Loss
    return (-1 / m) * (np.dot(np.log(A), Y.T)) + np.dot(np.log(1 - A), 1-Y.T) 