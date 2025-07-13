import matplotlib.pyplot as plt
from .neural_network import *
from .dataset import *

LABEL_SIZE = 10
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.1
DIMENSIONS = [784, 128, 64, 10]

learning_rates = [0.12, 0.1, 0.08]
epochs = range(10, 110, 10)

def main():
    accuracy_scores = []

    ds_train, ds_test = retrieve_mnist()
    x_train, y_train = clean_train(ds_train, BATCH_SIZE, LABEL_SIZE)
    x_test, y_test = clean_test(ds_test, BATCH_SIZE)

    for learning_rate in learning_rates:
        temp = []
        for epoch in epochs:
            parameters, cost_history = train(x_train, y_train, DIMENSIONS, epoch, learning_rate)
            accuracy_score = test(x_test, y_test, parameters)
            temp.append(accuracy_score)
            print(f"Epoch - {epoch}, Learning Rate - {learning_rate}: {accuracy_score}")

        accuracy_scores.append(temp)
    return accuracy_scores

def plot(accuracy_scores):
    for learning_rate, accuracy_score in zip(learning_rates, accuracy_scores):

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, accuracy_score)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Performance against Epochs @ {learning_rate} learning rate')
        plt.show()

if __name__ == "__main__":
    scores = main()
    plot(scores)