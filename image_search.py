# Code for detecting numbers and generating sudoku board

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


class ImageSearch:
    def __init__(self, image=None):

        self.coordinates = None
        self.lines = None
        self.image_matrix = None
        self.output = None
        if image:
            self.image = image
        else:
            path = os.path.join(os.getcwd(), 'images', 'sudoku_1.jpg')
            img = cv2.imread(path)
            self.image = img

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 90,150,apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

        if not lines.any():
            print('No lines were found')
            exit()

        if filter:
            rho_threshold = 15
            theta_threshold = 0.1

            # how many lines are similar to a given one
            similar_lines = {i: [] for i in range(len(lines))}
            for i in range(len(lines)):
                for j in range(len(lines)):
                    if i == j:
                        continue

                    rho_i, theta_i = lines[i][0]
                    rho_j, theta_j = lines[j][0]
                    if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                        similar_lines[i].append(j)

            # Ordering the INDICES of the lines by how many are similar to them
            indices = [i for i in range(len(lines))]
            indices.sort(key=lambda x: len(similar_lines[x]))

            # Line flags are the base for the filtering
            line_flags = len(lines)*[True]
            for i in range(len(lines) - 1):
                if not line_flags[indices[i]]:
                    continue

                for j in range(i + 1, len(lines)):  # We're considering elements that had lower amount of similar lines
                    if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                        continue

                    rho_i, theta_i = lines[indices[i]][0]
                    rho_j, theta_j = lines[indices[j]][0]
                    if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                        line_flags[indices[j]] = False  # If it's similar and has not been disregarded then drop it

        # print('number of Hough lines:', len(lines))

        filtered_lines = lines

        for line in filtered_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # print(filtered_lines)
        # cv2.imshow('hough.jpg', img)
        # cv2.waitKey(0)
        self.lines = filtered_lines
        self.gridsearch()
        self.crop_image()
        self.process_image()

    def gridsearch(self):
        lines = self.lines
        sorted_lines = np.vstack([x[0] for x in lines])
        sorted_lines = sorted_lines[sorted_lines[:, 0].argsort()]
        sorted_lines = sorted_lines[sorted_lines[:, 1].argsort(kind='mergesort')]
        grid_size = len(sorted_lines) // 2
        horiz_coord = np.array([x[0] for x in sorted_lines[0:grid_size]])
        vert_coord = np.array([x[0] for x in sorted_lines[grid_size:]])
        self.coordinates = [horiz_coord, vert_coord]
        # print(horiz_coord)
        # print(vert_coord)

    def crop_image(self):
        image = self.image
        Hcoord = list(map(int, self.coordinates[0]))
        Vcoord = list(map(int, self.coordinates[1]))
        image_matrix = []
        for j in range(len(Vcoord)-1):
            for i in range(len(Hcoord)-1):
                new_image = image[Vcoord[j]:Vcoord[j+1], Hcoord[i]:Hcoord[i+1]]
                image_matrix.append(new_image)
        self.image_matrix = image_matrix
        # cv2.imshow('Cropped Image', image_matrix[1])
        # cv2.waitKey(0)

    def process_image(self):
        image_matrix = self.image_matrix
        new_image_matrix = np.array(image_matrix, dtype=object)
        bw_image_matrix = []
        for i in range(new_image_matrix.size):
            bw_image = cv2.cvtColor(image_matrix[i], cv2.COLOR_BGR2GRAY)
            bw_image_matrix.append(bw_image)
        bw_image_matrix = np.array(bw_image_matrix, dtype=object)
        self.output = bw_image_matrix
        # bw_image_matrix = bw_image_matrix.reshape((9,9))
        # fig = plt.figure(figsize=(9, 9))
        # n = 1
        # for j in range(9):
        #     for i in range(9):
        #         fig.add_subplot(9,9,n)
        #         plt.imshow(bw_image_matrix[j][i])
        #         plt.axis('off')
        #         n += 1
        # plt.show()
        # cv2.imshow('Cropped Image', bw_image_matrix)
        # cv2.waitKey(0)
        r = 1





