import numpy as np

from .neural_network import *
from .dataset import *
from .utils import learning_rate_finder, plot

LABEL_SIZE = 10
BATCH_SIZE = 32
EPOCH = 200
LEARNING_RATE = 0.1
DIMENSIONS = [784, 128, 64, 10]
MODEL_PATH = './src/neural_network/params.pkl'

def find_learning_rate():
    losses = []
    learning_rates = []
    total = 100

    # Load Datasets
    ds_train, ds_test = retrieve_mnist()
    x_train, y_train = clean_train(ds_train, BATCH_SIZE, LABEL_SIZE)

    for step in range(0, total):
        # Find the learning rate
        learning_rate = learning_rate_finder(step, total, end=10)

        # Train the model
        cost_history = train(x_train, y_train, DIMENSIONS, EPOCH, learning_rate, MODEL_PATH, False, True)
        mean_loss = np.mean(cost_history)

        # Log
        print(f"Learning Rate - {learning_rate}: {mean_loss}")

        # Record for plotting
        learning_rates.append(learning_rate)
        losses.append(mean_loss)

    plot(learning_rates, losses, "Learning Rates", "Losses", "Losses against Learning Rates")

def find_epoch():
    losses = []

    # Load Datasets
    ds_train, ds_test = retrieve_mnist()
    x_train, y_train = clean_train(ds_train, BATCH_SIZE, LABEL_SIZE)

    for epoch in range(10, 210, 10):
        # Train the model
        cost_history = train(x_train, y_train, DIMENSIONS, epoch, LEARNING_RATE, MODEL_PATH, False, True)
        mean_loss = np.mean(cost_history)

        # Log
        print(f"Epoch - {epoch}: {mean_loss}")

        # Record for plotting
        losses.append(mean_loss)

    epochs = range(10, 210, 10)
    plot(epochs, losses, "Epochs", "Losses", "Losses against Epoch @ 0.1 Learning Rate")

def find_stability():
    # Load Datasets
    ds_train, ds_test = retrieve_mnist()
    x_train, y_train = clean_train(ds_train, BATCH_SIZE, LABEL_SIZE)

    # Train the model
    cost_history = train(x_train, y_train, DIMENSIONS, EPOCH, LEARNING_RATE, MODEL_PATH, False, False)

    cost_history_per_epochs = [np.mean(np.array(cost_history[start: start + len(ds_train) // BATCH_SIZE])) for start in range(0, len(cost_history), len(ds_train) // BATCH_SIZE)]
    epochs = range(0, EPOCH)

    plot(epochs, cost_history_per_epochs, "Epochs", "Cost History", "Cost History against Epochs")

def main():
    # Load Datasets
    ds_train, ds_test = retrieve_mnist()
    x_train, y_train = clean_train(ds_train, BATCH_SIZE, LABEL_SIZE)
    x_test, y_test = clean_test(ds_test, BATCH_SIZE)

    # Train the model
    train(x_train, y_train, DIMENSIONS, EPOCH, LEARNING_RATE, MODEL_PATH, False, True)

    # Test the model
    accuracy_score = test(x_test, y_test, MODEL_PATH)

    # Log
    print(f"{accuracy_score}")

    return accuracy_score

if __name__ == "__main__":
    score = main()