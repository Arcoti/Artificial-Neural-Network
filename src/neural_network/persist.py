import pickle

def save_model(parameters: dict, filepath: str):
    with open(filepath, 'wb') as file:
        pickle.dump(parameters, file)

def load_model(filepath: str):
    with open(filepath, 'rb') as file:
        return pickle.load(file)
    