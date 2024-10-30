# Neural Network from Scratch
This project demonstrates the implementation of a simple neural network from scratch using NumPy and relevant mathematical operations such as linear algebra and calculus. The goal is to classify handwritten digits from the MNIST dataset. I only used Keras to load the dataset; no other machine learning frameworks (like TensorFlow or PyTorch) were used.

## Project Overview
The notebook walks through the following steps:

### Data Loading:
The MNIST dataset is loaded using keras.datasets.
The pixel values are normalized by dividing by 255.
### Neural Network Implementation:
A neural network is built from scratch using NumPy that supports a network with two hidden layers. I chose to use the following network: 784, 128, 64, 10.
The common training procedure for a neural network is used: forward propagration, backpropagation, and gradient descent.
### Training and Evaluation:
The network is trained on the training set and is ran for 500 epochs.
### Visualization:
Random samples from what the model predicted of test dataset are visualized to show how well the model works.
## Prerequisites
To run this notebook, you will need:

* Python 3.x

The following libraries:

* numpy

* matplotlib

* keras

You can install the necessary libraries using:

```
pip install numpy matplotlib keras
```


## Instructions
Clone or download this repository to your local machine.
Open the notebook (nn_from_scratch.ipynb) in Jupyter Notebook or Jupyter Lab.
You can modify parameters such as the number of layers, neurons, epochs, or learning rate to experiment with different configurations.
## Results
The final trained model achieves an accuracy of 91% on the training and testing dataset. Further improvements can be made by optimizing the number of epochs, batch size, or implementing more complex architectures.

## Acknowledgements
The MNIST dataset is provided by the Keras library.
This project is inspired by wanting to understand how neural networks function.
