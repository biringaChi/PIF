__author__ = "biringaChi"

import os
import numpy as np
from skimage.measure import compare_ssim
import imageio
from skimage.transform import resize

def retrieve_img(path, norm_size=True):
    h = 2 ** 10
    w = 2 ** 10
    img = imageio.imread(path).astype(int)
    if norm_size:
        img = resize(img, (h, w), anti_aliasing=True, preserve_range=True)
    return img


def mse(path_a, path_b):
    img_a = retrieve_img(path_a)
    img_b = retrieve_img(path_b)
    if len(img_a) != len(img_b) and len(img_a[0]) != len(img_b[0]):
        raise Exception("Image dimensions are different")
    else:
        dist_measure = np.sum((img_a - img_b) ** 2)
        dist_measure /= float(img_a.shape[0] * img_a.shape[1])
        return dist_measure


def ssm(path_a, path_b):
    img_a = retrieve_img(path_a)
    img_b = retrieve_img(path_b)
    if len(img_a) != len(img_b) and len(img_a[0]) != len(img_b[0]):
        raise Exception("Image dimensions are different")
    else:
        sim, diff = compare_ssim(img_a, img_b, full=True, multichannel=True)
        return sim


def groupings(path_a, path_b):
    """group images based on similarity measure"""
    if(ssm(path_a, path_b) == 1):
        print("Perfect similarity score")
        dir_root = "similar"
        path = os.path.join("experiment", dir_root)
        os.mkdir(path)


if __name__ == '__main__':
    img_a = "experiment/frame0.jpg"
    img_b = "experiment/frame0copy.jpg"
    #mse = mse(img_a, img_b)
    #ssm = ssm(img_a, img_b)
    groupings(img_a, img_b)
    #print(mse)
    #print(ssm)
