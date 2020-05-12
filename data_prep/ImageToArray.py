import cv2

#the number of frames in each video
aerialFootage = 3722
planetEarth = 3274
popeye = 10979

def ImageToArrayColor(imagePath):
    #converts an image to a color ndarray
    
    im = cv2.imread(imagePath)
    
    return im


def ImageToArrayBW(imagePath):
    #converts an image to a grayscale ndarray
    
    im = cv2.imread(imagePath,mode=0)
    
    return im