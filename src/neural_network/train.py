from .function import cost_function
from .params import initialize_parameters, update_parameters
from .propagation import forward_propagation, backward_propagation
from .persist import load_model, save_model

def train(X: list, Y: list, layer_dims: list, epochs: int, learning_rate: float, model_path: str = "", load_existing: bool = False, save: bool = True):

    # Fetch Parameters
    if load_existing and model_path != "":
        parameters = load_model(model_path)
    else:
        parameters = initialize_parameters(layer_dims)

    # Initialize cost history list
    cost_history = []

    for i in range(epochs):
        for batch_X, batch_Y in zip(X, Y):
            # Pass the data forward
            Y_hat, caches = forward_propagation(batch_X, parameters)

            # Obtain the cost
            cost = cost_function(Y_hat, batch_Y)
            cost_history.append(cost)

            # Learn by back propagation
            gradients = backward_propagation(Y_hat, batch_Y, caches)
            parameters = update_parameters(parameters, gradients, learning_rate)

    # Save the model upon completion
    if save and model_path != "":
        save_model(parameters, model_path)
    
    # Return the cost_history for reference
    return cost_history
