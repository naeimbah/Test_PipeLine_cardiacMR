import os
import cv2
import numpy as np
import pandas as pd
from src.utility import mask_gen,pat_lst_gen

'''
contour_dir = '/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/contourfiles/'
contours = '/o-contours'
dicom_dir = '/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/dicoms/'
'''


def otsu_thr(img_otsu):
    """Generate three sets of thresholded masks based on 1) global thresholding 2) regular otsu's thresholding and 3) otsu's thresholding with gaussian filtering
    :param dir: list of dicoms and their corresponding contours.
    """
    img_otsu= np.array(img_otsu, dtype=np.uint8)

    # global thresholding
    ret1,th1 = cv2.threshold(img_otsu,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img_otsu,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_otsu,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return th1,th2,th3

# extract the look up table
lookup_table = pd.read_csv('/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/link.csv')

# generate the list of contours
dicom_sub_dirs_lst,contour_sub_dirs_lst = pat_lst_gen(dicom_dir,lookup_table)

# get the masks outer, inner, and ring
img, o_contour_out,i_contour_out,ring =  mask_gen(dicom_dir,contour_dir,dicom_sub_dirs_lst, contour_sub_dirs_lst)

# multiply the outer mask to the image
outter_heart = np.multiply(img,o_contour_out)

# now let's binarize the blood pool (th3) from outer contour
th1,th2,th3 = otsu_thr(img_otsu = outter_heart)
