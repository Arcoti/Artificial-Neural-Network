import numpy as np

from .function import sigmoid, sigmoid_first_derivative

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
        Z = np.dot(A_prev, W) + b

        # Store Linear Cache
        linear_cache = (A_prev, W, b)

        # Apply Activation Function
        A = sigmoid(Z)

        # Store Activation Cache
        activation_cache = Z

        cache = (linear_cache, activation_cache)
        caches.append(cache)
    
    return A, caches

def one_layer_back_propagation(dA, cache: tuple):
    linear_cache, activation_cache = cache

    Z = activation_cache
    dZ = dA * sigmoid_first_derivative(Z) 

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(A_prev.T, dZ)
    db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
    db = db.reshape(db.shape[1],)
    dA_prev = np.dot(dZ, W.T)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    gradients = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL)) - np.divide(1-Y, 1-AL)

    current_cache = caches[L - 1]
    gradients['dA'+str(L-1)], gradients['dW'+str(L-1)], gradients['db'+str(L-1)] = one_layer_back_propagation(dAL, current_cache)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_layer_back_propagation(gradients['dA'+str(l+1)], current_cache)
        gradients['dA' + str(l)] = dA_prev_temp
        gradients['dW' + str(l + 1)] = dW_temp
        gradients['db' + str(l + 1)] = db_temp
    
    return gradients
