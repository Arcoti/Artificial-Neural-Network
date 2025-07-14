# Artificial-Neural-Network
This project aims to create an Artificial Neural Network (ANN) from scratch which will be trained to identify handwritten numbers from the MNIST database. As an extension, the program is also able to take in an image and provide its prediction. 

### Overview

As a brief overview of the ANN of this project, the activation function used are Rectified Linear Unit (ReLU) for the hidden layers and Softmax function for the output layer. The Softmax function is specifically suited for the MNIST dataset since it considers the output in the last layer as a group rather than independently like the Sigmoid function. The network has a back propagation algorithm for it to learn and update its weights and biases. In addition, the loss is calculated by Categorical Cross Entropy. 

For the parameters, the ANN is set to have four layers with two hidden ones. The input layer possess 784 (28 by 28) neurons and the output layer has 10. The hidden layers possess 128 and 64 neurons respectively. The batch size used to train this network is 32. To determine the ideal learning rate, a graph of loss against learning rates at 2 epochs is plotted, as shown below. The learning rate of 0.1 is selected due to its low losses. While the higher learning rate results in lower losses, they are not selected due to fear of unstable solution. 

![Graph of Loss against Learning Rates @ 2 Epochs](./static/Graph%20-%20Loss%20vs%20Learning%20Rates%20@%202%20Epoch.png)

To determine the ideal epochs, a graph of loss against epochs at 0.1 learning rate is plotted, as shown below. The epochs of 200 is selected since it provides the lowest losses. 

![Graph of Loss against Epochs @ 0.1 Learning Rate](./static/Graph%20-%20Loss%20vs%20Epochs%20@%200.1%20Learning%20Rate.png)

The [stored model](./src/neural_network/params.pkl) is trained with a learning rate of 0.1, 200 epochs and batch size of 32. It is recorded to have a test accuracy score of 0.9765 when tested with a test dataset. 

A graph of how the model's losses improves after 20 epochs of training data is show below. 

![Graph of Loss against Epochs @ 0.1 Learning Rate for 20 Epochs](./static/Graph%20-%20Loss%20History%20vs%20Epochs%20@%200.01%20Learning%20Rate%20and%2020%20Epochs.png)

As an extension, the program can take in a [raw input image](./static/Sample_1.png) and accurately provide predictions on its digits. 

### Running and Installation

1. Set up and activate the Python Virtual Environment
```
python -m venv .venv
.venv\Scripts\activate
```

2. Install pdm
```
pip install pdm
```

3. Install dependencies
```
pdm sync
```

3. Run the code
```
python -m src.main
```

### Editing the Code

Install packages directly using pdm
```
pdm add <package-name>
```

Or, install via pip and add them
```
pip install <package-name>
pdm add requests <package-name>
```

### Acknowledgements
I would like to express my thanks to the following resources:

- [Neural Networks by 3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks)
- [How to build Neural Network from Scratch](https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/)
