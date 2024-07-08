import numpy as np
from skimage import img_as_ubyte
from skimage.color import rgb2lab, rgb2gray
from skimage.transform import resize
from skimage.filters import roberts, sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2

def extract_custom_features(img):
    if img.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels (RGB)")

    LAB_img = rgb2lab(img)
    A_img = LAB_img[:,:,1]
    A_feat = A_img.mean()
    B_img = LAB_img[:,:,2]
    B_feat = B_img.mean()
    
    gray_img = rgb2gray(img)
    gray_img = resize(gray_img, (256, 256))
    gray_img = img_as_ubyte(gray_img)
    
    entropy_img = entropy(gray_img, disk(3))
    entropy_mean = entropy_img.mean()
    entropy_std = entropy_img.std()
    
    roberts_img = roberts(gray_img)
    roberts_mean = roberts_img.mean()

    sobel_img = sobel(gray_img)
    sobel_mean = sobel_img.mean()
    
    kernel1 = cv2.getGaborKernel((9, 9), 3, np.pi/4, np.pi, 0.5, 0, ktype=cv2.CV_32F)
    gabor1 = (cv2.filter2D(gray_img, cv2.CV_8UC3, kernel1)).mean()
    
    kernel2 = cv2.getGaborKernel((9, 9), 3, np.pi/2, np.pi/4, 0.9, 0, ktype=cv2.CV_32F)
    gabor2 = (cv2.filter2D(gray_img, cv2.CV_8UC3, kernel2)).mean()

    kernel3 = cv2.getGaborKernel((9, 9), 5, np.pi/2, np.pi/2, 0.1, 0, ktype=cv2.CV_32F)
    gabor3 = (cv2.filter2D(gray_img, cv2.CV_8UC3, kernel3)).mean()

    custom_features = np.array([A_feat, B_feat, entropy_mean, entropy_std, roberts_mean, 
                                sobel_mean, gabor1, gabor2, gabor3])
    
    return custom_features

