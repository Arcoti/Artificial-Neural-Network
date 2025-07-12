import numpy as np

def initialize_parameters(layer_dims: list[int]):
    """
    Initialize weights and biases for all the different layers

    Parameters
    ----------
    layer_dims : list[int]
        The dimensions of each layer. E.g. [input_size, hidden_1, hidden2, ..., output_size]
    """

    # Define seed so that every run results in the same parameters
    np.random.seed(1)

    parameters = {}

    # Generate the weights (W) and biases (b)
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l - 1], layer_dims[l])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],))

    return parameters

def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * gradients['db' + str(l)]

    return parameters