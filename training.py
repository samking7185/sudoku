from image_NN import Network
import numpy as np
import load_mnist

training_data, validation_data, test_data = load_mnist.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)

net = Network([784, 60, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

np.savetxt('Biases1.txt', net.biases[0])
np.savetxt('Biases2.txt', net.biases[1])
np.savetxt('Biases3.txt', net.biases[2])

np.savetxt('Weights1.txt', net.weights[0])
np.savetxt('Weights2.txt', net.weights[1])
np.savetxt('Weights3.txt', net.weights[2])

# textfile = open("results.txt", "w")
# iteration = 1
#
# for layer in net.biases:
#     sec_title = "________ Bias Layer %d ________" % iteration
#     textfile.write(sec_title + "\n")
#     iteration += 1
#     for neuron in layer:
#         textfile.write(str(neuron) + "\n")
#
# iteration = 1
# for layer in net.weights:
#     sec_title = "________ Weights Layer %d ________" % iteration
#     textfile.write(sec_title + "\n")
#     iteration += 1
#     for neuron in layer:
#         textfile.write(str(neuron) + "\n")
# textfile.close()