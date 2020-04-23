__author__ = "biringaChi"

import numpy as np

class PerformanceMeasures():
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def pixel_distance(self):
        if len(self.img1) != len(self.img2) and len(self.img1[0]) != len(self.img2[0]):
            raise Exception("Image dimensions are different")
        else:
            dist_measure = np.sum((self.img1 - self.img2) ** 2)
            dist_measure /= float(self.img1.shape[0] * self.img1.shape[1])
            return dist_measure


if __name__ == '__main__':
    pd = PerformanceMeasures()
