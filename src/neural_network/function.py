import numpy as np

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_first_derivative(Z: np.ndarray):
    return (Z > 0).astype(float)

def softmax(Z):
    exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

def sigmoid(Z):
    return 1 / (1 + np.exp(np.dot(-1, Z)))

def sigmoid_first_derivative(Z: np.ndarray):
    return sigmoid(Z) * (1-sigmoid(Z))

# def cost_function(A: np.ndarray, Y: np.ndarray):
#     m = Y.shape[1]

#     # Binary Cross Entropy Loss
#     return (-1 / m) * (np.dot(np.log(A), Y.T)) + np.dot(np.log(1 - A), 1-Y.T) 

def cost_function(A: np.ndarray, Y: np.ndarray, epsilon: float = 1e-08):
    m = Y.shape[1]

    # Categorical Cross Entropy
    return -np.sum(Y * np.log(A + epsilon)) / m