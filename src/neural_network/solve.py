from .propagation import forward_propagation

def solve(X, parameters):
    A, caches = forward_propagation(X, parameters)
    return A
