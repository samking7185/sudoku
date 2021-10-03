import numpy as np
import load_mnist
from features import *

training_data, validation_data, test_data = load_mnist.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)

image = training_data[15][0]
image_size = [20,20]
new_image = extraction(image, image_size)
# gradI = transition(new_image)
skel = skel(new_image)
# This is the training for the neural network
# net = Network([784, 60, 30, 10])
#
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#
# np.savetxt('Biases1.txt', net.biases[0])
# np.savetxt('Biases2.txt', net.biases[1])
# np.savetxt('Biases3.txt', net.biases[2])
#
# np.savetxt('Weights1.txt', net.weights[0])
# np.savetxt('Weights2.txt', net.weights[1])
# np.savetxt('Weights3.txt', net.weights[2])

