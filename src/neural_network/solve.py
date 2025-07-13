import numpy as np
from .propagation import forward_propagation
from .persist import load_model

def solve(X, parameters: dict):                     # Use with test
    A, caches = forward_propagation(X, parameters)
    return interpret(A)

def interpret(A):
    return np.array([np.argmax(case) for case in A])

def predict(X, model_path: str):                    # Use alone
    parameters = load_model(model_path)
    return solve(X, parameters)
