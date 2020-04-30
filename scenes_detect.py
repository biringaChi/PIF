__author__ = 'JosueCom'
__date__ = '4/30/2020'
__email__ = "josue.n.rivera@outlook.com"

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from data_prepocessing.ImageToArray import ImageToArrayColor

ini = 1000
lst = 3000#10979
path = "data_prepocessing\\Popeye"
diff = []

img = ImageToArrayColor(os.path.join(path, "frame" + str(ini) + ".jpg"))
prev = np.sum(img)/(img.size)

for i in range(ini + 1, lst + 1):
	img = ImageToArrayColor(os.path.join(path, "frame" + str(i) + ".jpg"))

	if(np.sum(img) == None):
		print(os.path.join(path, "frame" + str(i) + ".jpg"))
		continue

	curr = np.sum(img)/(img.size)

	diff.append(curr - prev)
	prev = curr

	if(i%200 == 0):
		print(">" + str(i))

#print(diff)

plt.plot(list(range(len(diff))), diff, "b-")
plt.ylabel('difference')
plt.xlabel('scene')
plt.show()
