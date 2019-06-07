import os
import cv2
import numpy as np
import pandas as pd
from src.utility import mask_gen,pat_lst_gen,i_mask_gen
import sklearn
from sklearn.metrics import jaccard_similarity_score

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
outer_heart = np.multiply(img,o_contour_out)

# now let's binarize the blood pool (th3) from outer contour
th1,th2,th3 = otsu_thr(img_otsu = outer_heart)



def eval_seg(outer_heart,method):
    """create arrays of similarity measuerements for a give set of outer heart masks-- outer_heart
    :param dir: array of outer heart contours and method of evaluation, 'otsu' or others.
    """

    pred_blood_pool = []
    JSS = []
    dice_coef = []


    for i in range(len(outer_hear)):

        # this function is just defined for otsu or JSS measurements
        if method == 'otsu':
            th1,th2,pred = otsu_thr(outer_heart[i])
        else:
            # creating inner mask using morphological approach
            pred = i_mask_gen(outer_heart[i])

        pred[pred>0] = 1
        pred_blood_pool.append(pred)

        # grount truth
        GT = blood_pool[i]
        GT[GT>0] = 1
        print('jaccard similarity score is : {}'.format(sklearn.metrics.jaccard_similarity_score(pred, GT)))

        # get JSS score 
        JSS.append(sklearn.metrics.jaccard_similarity_score(pred, GT))

        # get dice score
        k = 1
        dice = np.sum(pred[GT==k])*2.0 / (np.sum(pred) + np.sum(GT))
        print('dice coefficient is : {}'.format(dice))
        dice_coef.append(dice)


    return JSS,dice_coef
