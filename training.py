from image_NN import Network
import numpy as np
import load_mnist

training_data, validation_data, test_data = load_mnist.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)

net = Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

textfile = open("results.txt", "w")
iteration = 1

for layer in net.biases:
    sec_title = "________ Bias Layer %d ________" % iteration
    textfile.write(sec_title + "\n")
    iteration += 1
    for neuron in layer:
        textfile.write(str(neuron) + "\n")

iteration = 1
for layer in net.weights:
    sec_title = "________ Weights Layer %d ________" % iteration
    textfile.write(sec_title + "\n")
    iteration += 1
    for neuron in layer:
        textfile.write(str(neuron) + "\n")
textfile.close()