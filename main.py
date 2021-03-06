# This is the main run script for the Sudoku  Solver
import numpy as np
from solver import Solver
from read_variables import OpenFile
from apply import ApplyNetwork
from image_search import ImageSearch
import load_mnist
import cv2

grid = [
        [0, 8, 4, 0, 2, 9, 5, 0, 1],
        [5, 7, 0, 0, 0, 1, 0, 3, 0],
        [6, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 5, 0, 0, 9, 6, 0],
        [0, 0, 0, 0, 3, 0, 1, 0, 4],
        [2, 0, 6, 9, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0, 0, 0, 0],
        [7, 0, 0, 0, 0, 3, 0, 0, 6],
        [0, 0, 1, 0, 0, 0, 0, 0, 0]
]
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
# cv2.imshow('hough.jpg', np.reshape(picture, (28,28)))
# cv2.waitKey(0)

network_size = [784, 60, 30, 10]
files = ["Biases1.txt", "Biases2.txt", "Biases3.txt", "Weights1.txt", "Weights2.txt", "Weights3.txt"]
data = OpenFile(files)

squares = ImageSearch()
neural_net = ApplyNetwork(network_size, data.weights, data.biases)
nums = []
# output = neural_net.calculate(picture)

images = squares.output[0]
empties = squares.output[1]
outs = []
for e, pic in zip(empties, images):
        if e == 0:
                nums.append(0)
                outs.append([0])
        else:
                pic = pic.flatten()
                output = neural_net.calculate(pic)
                nums.append(np.argmax(neural_net.output))
                outs.append(neural_net.output)

# Code to print actual vs. detected image
# print('------------------------')
# print('Original Grid')
# print('------------------------')
# print(np.array(grid))
# print('------------------------')
#
# print('Detected Grid')
# print('------------------------')
#
# print(np.reshape(nums, (9, 9)))

grid = np.reshape(nums, (9, 9))
solution = Solver(grid.tolist())
print('-------SOLVED--------')
print(np.matrix(solution.grid))
