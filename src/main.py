from .neural_network import *
from .dataset import *

LABEL_SIZE = 10
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.1
DIMENSIONS = [784, 128, 64, 10]

def main():
    ds_train, ds_test = retrieve_mnist()
    x_train, y_train = clean_train(ds_train, BATCH_SIZE, LABEL_SIZE)
    x_test, y_test = clean_test(ds_test, BATCH_SIZE)

    parameters, cost_history = train(x_train, y_train, DIMENSIONS, EPOCHS, LEARNING_RATE)
    print(parameters)

    accuracy_score = test(x_test, y_test, parameters)
    print(accuracy_score)

if __name__ == "__main__":
    main()