from .function import cost_function
from .params import initialize_parameters, update_parameters
from .propagation import forward_propagation, backward_propagation

def train(X, Y, layer_dims, epochs, learning_rate):
    parameters = initialize_parameters(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_propagation(X, parameters)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)

        gradients = backward_propagation(Y_hat, Y, caches)
        parameters = update_parameters(parameters, gradients, learning_rate)
    
    return parameters, cost_history
