from __future__ import print_function
import torch
import torch.nn as nn
import pickle
import data_prep as prep
from torchvision import transforms, utils
import numpy as np
from generator import Generator
from torch.utils.data import DataLoader
import cv2
from data_prepocessing.ImageToArray import ImageToArrayColor
import os

__author__ = 'JosueCom'
__date__ = '5/8/2020'
__email__ = "josue.n.rivera@outlook.com"


#image_size = 500
batch_size = 4
image_size = 500
ini = 0
lst = 3274#10979
from_path = "data_prepocessing/PlanetEarth"
to_path = "out"
count = 0

permute = [2, 1, 0]

diff_pickle = open("planet_earth_diff.pickle","rb")

tf = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

netG = torch.load("nice/generator6.bin")
netG.eval()

print("running model")

prev = tf(ImageToArrayColor(os.path.join(from_path, "frame" + str(ini) + ".jpg")))
utils.save_image(prev[permute, :, :], os.path.join(to_path, "frame" + str(count) + ".jpg"), normalize=True, padding=0)

for i in range(ini + 1, lst + 1):
    curr = tf(ImageToArrayColor(os.path.join(from_path, "frame" + str(i) + ".jpg")))
    
    torch.cuda.empty_cache()
    out = netG(prev.to(device).unsqueeze(0), curr.to(device).unsqueeze(0))
    og = out[0][permute, :, :]
    
    utils.save_image(og, os.path.join(to_path, "frame" + str(count + 1) + ".jpg"), normalize=True, padding=0)
    utils.save_image(curr[permute, :, :], os.path.join(to_path, "frame" + str(count + 2) + ".jpg"), normalize=True, padding=0)
    count += 2
    prev = curr
    
    if(i%20 == 1):
        print(str(i) + "/" + str(lst+1))

print("Done running model")