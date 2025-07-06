# Artificial-Neural-Network
This project aims to create an Artificial Neural Network (ANN) from scratch which will be trained to identify handwritten numbers from the MNIST database. 

### Overview

As a brief overview, the ANN of this project possess one input layer, two hidden layers and one output layers. Ideally, the input layer will have 784 neurons (28 by 28) and the output layer will have 10 neurons, one for each digit. The hidden layers will have varying number of neurons with 128 neurons for the first and 64 neurons for the second. To produce a non-linear function, the Rectified Linear Unit (ReLU) will be used over the sigmoid function. The ANN will basically adopt backpropagation to tune their weights and bias for each neuron to produce the ideal reasults. 

### Running and Installation

1. Set up and activate the Python Virtual Environment
```
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies
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
