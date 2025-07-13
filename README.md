# Artificial-Neural-Network
This project aims to create an Artificial Neural Network (ANN) from scratch which will be trained to identify handwritten numbers from the MNIST database. 

### Overview

As a brief overview of the ANN of this project, the activation function used are Rectified Linear Unit (ReLU) for the hidden layers and Softmax function for the output layer. The Softmax function is specifically suited for the MNIST dataset since it considers the output in the last layer as a group rather than independently like the Sigmoid function. The network has a back propagation algorithm for it to learn and update its weights and biases. In addition, the loss is calculated by Categorical Cross Entropy. 

For the parameters, the ANN is set to have four layers with two hidden ones. The input layer possess 784 (28 by 28) neurons and the output layer has 10. The hidden layers possess 128 and 64 neurons respectively. The batch size used to train this network is 32. After conducting a series of tests as shown by the graphs below, it is found that __ epochs and __ learning rate produces the highest accuracy score. As such, these parameters are used for the model. 

For future extension, can try to make the model clean raw input data like an image, turn it into greyscale and pass it into the model for analysis. 

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
- [Building Convolutional Neural Network using NumPy from Scratch](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)
