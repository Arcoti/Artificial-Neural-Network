import numpy as np
from .propagation import forward_propagation

def solve(X, parameters):
    A, caches = forward_propagation(X, parameters)
    return interpret(A)

def interpret(A):
    return np.array([np.argmax(case) for case in A])
