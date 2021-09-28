"""
~~~~~~~~~~~~
Variable Reader
~~~~~~~~~~~~
This program reads in the weights written to text files during training of an number recognition neural network
This reads in list of files then returns an object with the data
"""
import numpy as np


class OpenFile:
    def __init__(self, files):
        self.files = files
        self.biases = []
        self.weights = []
        for file in files:
            self.readfile(file)

    def readfile(self, file):
        data = np.loadtxt(file)
        if "Biases" in file:
            self.biases.append(data)
        else:
            self.weights.append(data)


