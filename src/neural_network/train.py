import numpy as np
from .function import cost_function
from .params import initialize_parameters, update_parameters
from .propagation import forward_propagation, backward_propagation

def train(X, Y, layer_dims, epochs, learning_rate):
    parameters = initialize_parameters(layer_dims)
    cost_history = []

    for i in range(epochs):
        for batch_X, batch_Y in zip(X, Y):
            Y_hat, caches = forward_propagation(batch_X, parameters)
            cost = cost_function(Y_hat, batch_Y)
            cost_history.append(cost)

            gradients = backward_propagation(Y_hat, batch_Y, caches)
            parameters = update_parameters(parameters, gradients, learning_rate)
    
    return parameters, cost_history
