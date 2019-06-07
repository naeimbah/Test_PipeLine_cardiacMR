import os
import cv2
import numpy as np
import pandas as pd
import sklearn
from scipy import ndimage

# calculating the gradients

def gaussian_kernel(size, sigma=1):
    """smooth the image
    :param dir: size and sigma for the guassian.
    """

    # guassian equation
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    """sobel filter to extract the gradients from an image
    :param dir: a 2D image in numpy array
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

# non maximum suppression
def non_max_suppression(img, D):
    """Removing the non maximum intensity to thinning the edges of an image
    :param dir: a 2D image in numpy array and the angle of thinning
    """
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)

    # get the angle of suppression
    angle = D * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

# double thresholding
def threshold(img, lowThresholdRatio, highThresholdRatio):
    """Generate the low and high threshold in an image
    :param dir: a 2D image in numpy array and ratios for low and high threshold
    """

    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    #thresholding
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


# Edge Tracking by Hysteresis
def hysteresis(img, weak, strong=255):
    """Edge tracking using hysteresis
    :param dir: a 2D image in numpy array and low and high thresholds extracted from threshold function
    """

    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def gen_vec(mask):
    """vectorize a mask 
    :param dir: 2D mask
    """
   x,y = np.nonzero(mask)
   vector = mask[x,y]
   return vector



def i_mask_gen(outer_heart_mask):
    """predicting the inner mask form the outer heart mask
    :param dir: outer heart mask
    """

    # get the gradients
    G, theta = sobel_filters(outer_heart_mask)

    # suppress the non maximum intensities
    supressed = non_max_suppression(G, theta)
    supressed[supressed < 50] = 0

    # double thresholding
    res, weak, strong = threshold(supressed, lowThresholdRatio=0.05, highThresholdRatio=0.09)

    # edge tracking with hystersis
    Hysteresis = hysteresis(res, weak, strong=255)
    Hysteresis_1 = np.array(Hysteresis, dtype=np.uint8)

    # Bolding the edges
    des = cv2.bitwise_not(Hysteresis_1)
    contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(des,[cnt],0,255,-1)
    gray = cv2.bitwise_not(des)

    # two level island removal
    # removing inner islands
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    gray = np.array(gray, dtype=np.uint8)
    mask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    out = Hysteresis * (mask/256)

    # removing outer islands
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    gray = np.array(out, dtype=np.uint8)
    mask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    out = Hysteresis * (mask/256)

    # filling the inner contour
    outer_heart_vec = gen_vec(outer_heart_mask)

    # kernel design
    size_d = int(np.sqrt(len(outer_heart_vec)))
    size_e = int(1.1*size_d)

    # dilating
    kernel = np.ones((size_d,size_d), np.uint8)
    d_im = cv2.dilate(out, kernel, iterations=1)

    # eroding
    kernel = np.ones((size_e,size_e), np.uint8)
    e_im = cv2.erode(d_im, kernel, iterations=1)

    return e_im
