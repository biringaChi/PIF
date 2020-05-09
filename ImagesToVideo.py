# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn= 'out/'
<<<<<<< HEAD
pathOut = 'videoRsult2.mp4'

fps = 50
ini = 0
lst = 6549#10979
frame_array = []
=======
pathOut = 'videoRsult.mp4'

fps = 50
>>>>>>> 5a5f5703cdf993ffdab858b25ba406c2a1b4c58e
"""frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
<<<<<<< HEAD
files.sort()
=======
files.sort()"""
frame_array = []
>>>>>>> 5a5f5703cdf993ffdab858b25ba406c2a1b4c58e
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