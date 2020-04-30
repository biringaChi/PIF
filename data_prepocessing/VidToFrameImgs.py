# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:06:07 2020

@author: rjsem
"""

# Program To Read video 
# and Extract Frames 
import cv2 

# Function to extract frames 
def FrameCapture(path): 
	
	# Path to video file 
    vidObj = cv2.VideoCapture(path) 

	# Used as counter variable 
    count = 0

	# checks whether frames were extracted 
    success = 1

    while success: 

		# vidObj object calls read 
		# function extract frames 
        success, image = vidObj.read() 

		# Saves the frames with frame-count 
        if success != 1: break
        
        cv2.imwrite("frame%d.jpg" % count, image)  

        count += 1
        
    vidObj.release()
    cv2.destroyAllWindows()
# Driver Code 
if __name__ == '__main__': 

	# Calling the function 
    FrameCapture("PlanetEarth\\PlanetEarth.mp4") 
