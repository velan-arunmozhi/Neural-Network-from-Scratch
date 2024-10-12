# Neural Network from Scratch
This project demonstrates the implementation of a simple neural network from scratch using NumPy and relevant mathematical operations such as linear algebra and calculus. The goal is to classify handwritten digits from the MNIST dataset. We only use Keras to load the dataset; no other machine learning frameworks (like TensorFlow or PyTorch) are involved in the model creation.

## Project Overview
The notebook walks through the following steps:

Data Loading:
The MNIST dataset, containing 28x28 grayscale images of digits, is loaded using keras.datasets.
The pixel values are normalized by dividing by 255.
Neural Network Implementation:
A neural network is built from scratch using NumPy, with layers and activation functions implemented manually.
Backpropagation and gradient descent are used for training the model.
Training and Evaluation:
The network is trained on the MNIST dataset and evaluated for accuracy.
Visualization:
Random samples from the dataset are visualized to provide a better understanding of the input.
## Prerequisites
To run this notebook, you will need:

Python 3.x
The following libraries:
numpy
matplotlib
keras
You can install the necessary libraries using:

bash
Copy code
pip install numpy matplotlib keras
## Instructions
Clone or download this repository to your local machine.
Open the notebook (nn_from_scratch.ipynb) in Jupyter Notebook or Jupyter Lab.
Run the cells step by step to see the neural network implementation in action.
You can modify parameters such as the number of layers, neurons, or learning rate to experiment with different configurations.
## Results
The final trained model achieves decent accuracy on the MNIST dataset, demonstrating how a neural network works under the hood without relying on high-level machine learning libraries. Further improvements can be made by optimizing the number of epochs, batch size, or implementing more complex architectures.

## Acknowledgements
The MNIST dataset is provided by the Keras library.
This project is inspired by the desire to understand how neural networks function at a fundamental level.
