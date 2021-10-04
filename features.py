import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.ndimage.morphology as morph
from fuzzy_struc import FIS

"""
features.py
~~~~~~~~~~~~
A library of methods for processing images and preparing them for a fuzzy image recognition system
"""


def extraction(image, size: list):
    """This method takes in an MNIST image, finds the bounds of the number itself, then crops and resizes
    the image to a known "size"
    """
    # MNIST images are flattened so we must reshape first
    image = np.reshape(image, (28, 28))

    # Find bounds of non zero elements in image
    coordinates = np.nonzero(image)
    column_min = int(np.min(coordinates[1]))
    column_max = int(np.max(coordinates[1]))

    row_min = int(np.min(coordinates[0]))
    row_max = int(np.max(coordinates[0]))

    if row_max - row_min > column_max - column_min:
        dif = ((row_max - row_min) - (column_max - column_min)) / 2
        off = dif % 2 % 1
        column_max = int(column_max + dif + off)
        column_min = int(column_min - dif + off)
    elif row_max - row_min < column_max - column_min:
        dif = abs((row_max - row_min) - (column_max - column_min)) / 2
        off = dif % 2 % 1
        column_max = int(column_max - dif + off)
        column_min = int(column_min + dif + off)

    # Crop Image around number
    crop_img = image[row_min:row_max, column_min:column_max]
    crop_size = [row_max - row_min, column_max - column_min]
    scale_factor = size[1] / crop_size[1]

    width = int(crop_img.shape[1] * scale_factor)
    height = int(crop_img.shape[0] * scale_factor)
    dim = (width, height)
    new_image = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

    return new_image


def transition(image):
    grad1 = np.array(np.gradient(image))
    grad2H = np.array(np.gradient(grad1[1]))
    grad2V = np.array(np.gradient(grad1[0]))
    grad2H = grad2H[1]
    grad2V = grad2V[0]


# def zero_crossing(arr):
#     idx = np.where(np.diff(np.sign(arr)))[0]
#     return idx


def skel(image):
    BW = cv2.threshold(image, 0.6, 1, cv2.THRESH_BINARY)
    BW = np.array(BW[1])
    M, N = BW.shape

    '''Calculating Euclidean Distance of the Binary Image'''
    D, IDX = morph.distance_transform_edt(BW, return_distances=True, return_indices=True)
    grad = np.array(np.gradient(D))
    edge = np.zeros_like(image)
    nonzeros = np.nonzero(D)
    rows = nonzeros[0]
    cols = nonzeros[1]

    parameters = {
        "MF1": [(0.1, 0)],
        "MF2": [(0.1, 0)],
        "OUT": [(0, 0, 0.7), (0.1, 1, 1)],
        "RULES": [0],
        "BOUNDS": [0, 1]
    }

    gFIS = FIS
    for row, col in zip(rows, cols):
        current = D[row, col]
        in1 = grad[0][row-1, col] + grad[0][row+1, col]
        in2 = grad[1][row, col-1] + grad[1][row, col+1]
        s =  1
    s = 1



