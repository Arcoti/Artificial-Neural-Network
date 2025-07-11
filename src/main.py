from .neural_network import *
from .dataset import *

def main():
    ds_train, ds_test = retrieve_mnist()
    x_train, y_train = clean(ds_train, as_label=False)
    x_test, y_test = clean(ds_test, as_label=True)

    parameters, cost_history = train(x_train, y_train, [784, 128, 64, 10], 2, 0.1, 32)
    print(parameters)
    # accuracy_score = test(x_test, y_test, parameters)
    # print(accuracy_score)

if __name__ == "__main__":
    main()