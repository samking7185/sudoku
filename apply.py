"""
~~~~~~~~~~~~
Apply Network Training
~~~~~~~~~~~~
This class reads in the weights and biases of a trained network and creates an object that can take in an input
and give a result based on the network training.  All the variables should be lists of Numpy arrays

input: Input to be evaluated by Network
dimensions: Size of Neural Network, [Input Layer, Layer2, Layer 3, ... , Output Layer]
weights: Weights of each layer of network, [W1, W2, ... , Wout]
biases: Biases of each layer of network, [B1, B2, ... , Bout]
"""
import numpy as np
import math


class ApplyNetwork:
    def __init__(self, dimensions, weights, biases):
        self.dims = dimensions
        self.weights = weights
        self.biases = biases
        self.output = None

    def calculate(self, inputs):
        dimensions = self.dims
        weights = self.weights
        biases = self.biases
        layers = []
        inputs = np.reshape(inputs, (len(inputs),))
        for i in range(len(dimensions) - 1):
            layer_list = np.zeros(dimensions[i+1])
            for idx, (w, b) in enumerate(zip(weights[i], biases[i])):
                layer = np.sum(w * inputs) + b
                layer_list[idx] = 1 / (1 + math.exp(float(-layer)))

            inputs = layer_list
        self.output = inputs




