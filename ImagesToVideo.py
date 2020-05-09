# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn= 'out/'
pathOut = 'videoRsult2.mp4'

fps = 50
ini = 0
lst = 6549#10979
frame_array = []
"""frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
files.sort()
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])"""

for i in range(ini, lst):
    filename=os.path.join(pathIn, "frame" + str(i) + ".jpg")
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    if i%20 == 0:
        print(str(i) + "/" + str(len(frame_array)))
    # writing to a image array
    out.write(frame_array[i])

out.release()