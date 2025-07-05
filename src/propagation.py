import numpy as np

from .function import ReLU, ReLU_first_derivative

def forward_propagation(X: np.ndarray, params: dict):
    # Define Local Variables
    A = X
    caches = []

    # Forward Propagation Loop
    for l in range(1, (len(params) // 2) + 1):
        A_prev = A
        W = params['W'+str(l)]
        b = params['b'+str(l)]

        # Linear Hypothesis
        Z = np.dot(W, A_prev) + b

        # Store Linear Cache
        linear_cache = (A_prev, W, b)

        # Apply Activation Function
        A = ReLU(Z)

        # Store Activation Cache
        activation_cache = Z

        cache = (linear_cache, activation_cache)
        caches.append(cache)
    
    return A, caches

def one_layer_propagation(dA, cache: tuple):
    linear_cache, activation_cache = cache

    # Assume g(x) = activation function
    # dZ = dA * g'(x)
    Z = activation_cache
    dZ = dA * ReLU_first_derivative(Z) 

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    gradients = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL)) - np.divide(1-Y, 1-AL)