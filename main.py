# This is the main run script for the Sudoku  Solver
import numpy as np
from solver import Solver
from read_variables import OpenFile
from apply import ApplyNetwork
from image_search import ImageSearch
import load_mnist
import cv2

# grid = [
#         [5, 3, 0, 0, 7, 0, 0, 0, 0],
#         [6, 0, 0, 1, 9, 5, 0, 0, 0],
#         [0, 9, 8, 0, 0, 0, 0, 6, 0],
#         [8, 0, 0, 0, 6, 0, 0, 0, 3],
#         [4, 0, 0, 8, 0, 3, 0, 0, 1],
#         [7, 0, 0, 0, 2, 0, 0, 0, 6],
#         [0, 6, 0, 0, 0, 0, 2, 8, 0],
#         [0, 0, 0, 4, 1, 9, 0, 0, 5],
#         [0, 0, 0, 0, 8, 0, 0, 7, 9]
# ]

# grid = [
#     [7, 8, 0, 4, 0, 0, 1, 2, 0],
#     [6, 0, 0, 0, 7, 5, 0, 0, 9],
#     [0, 0, 0, 6, 0, 1, 0, 7, 8],
#     [0, 0, 7, 0, 4, 0, 2, 6, 0],
#     [0, 0, 1, 0, 5, 0, 9, 3, 0],
#     [9, 0, 4, 0, 6, 0, 0, 0, 5],
#     [0, 7, 0, 3, 0, 0, 0, 1, 2],
#     [1, 2, 0, 0, 0, 7, 4, 0, 0],
#     [0, 4, 9, 2, 0, 6, 0, 0, 7]
# ]
# print(np.matrix(grid))
#
# solution = Solver(grid)
# print('-------SOLVED--------')
# print(np.matrix(solution.grid))

# training_data, validation_data, test_data = load_mnist.load_data_wrapper()
# training_data = list(training_data)
# picture = training_data[2]
# picture = picture[0]
#
# cv2.imshow('hough.jpg', np.reshape(picture, (28,28)))
# cv2.waitKey(0)
# network_size = [784, 30, 10]
# files = ["Biases1.txt", "Biases2.txt", "Weights1.txt", "Weights2.txt"]
# data = OpenFile(files)
#
# neural_net = ApplyNetwork(network_size, data.weights, data.biases)
# output = neural_net.calculate(picture)
# print(np.argmax(neural_net.output))
picture = ImageSearch()
