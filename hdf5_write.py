# create HDF5 to store the data and read them in the data generator
#!/usr/bin/env python

# load libraries
import os
import re
import h5py
import dicom
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import pydicom
from pydicom.errors import InvalidDicomError
import pandas as pd
from PIL import Image, ImageDraw
from src.utility import parse_contour_file,parse_dicom_file,poly_to_mask,save_ovelay_iamges

# directories
dicom_dir = '/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/dicoms/'
contour_dir = '/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/contourfiles/'
# load csv in pandas dataframe format
lookup_table = pd.read_csv('/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/link.csv')



# script level variables
path = dicom_dir
settings_dict = {
        "target_size": (256,256),
        "data_path": "/home/nabahrami/Documents/Naeim/PipeLine/data/data.hdf5"
                 }

# getting the list of labels, dicom images, and pateint ids from utility
label_path_lst,dicom_path_lst,image_id_lst = save_ovelay_iamges(contour_dir, dicom_dir,lookup_table)
key_lst = image_id_lst
key_lst = [x for y in key_lst for x in y]

image_path_lst = dicom_path_lst
label_path_lst = label_path_lst



def process_hdf5(image_path_lst,key_lst,label_path_lst, settings_dict):
    """ :param dir: list of the path to images , labels, and patient ids (i.e., keys)
        settings_dict:
            [target_size]: the target Y Z dimentions
            [data_path]: the path for the data path
    EFFECT:
        creates a HDF5 file in the base data dir
            *** WILL OVERWRITE OLD HDF5 FILE! ***
    """
    # verify settings_dict has correct keys
    settings_key_lst = ["target_size", "data_path"]
    for key in settings_key_lst:
        if not key in settings_dict:
            raise KeyError("{} not in settings_dict".format(key))

    # convert to tuple
    tuple_lst = list(zip(image_path_lst,label_path_lst,key_lst))

    # make file link
    data_file = h5py.File(settings_dict["data_path"])

    storage = []
    for curr_tuple in tqdm(tuple_lst):
        storage.append(process_dicom_series(settings_dict["target_size"],curr_tuple))

    print("\n")
    # remove None's
    storage = list(filter(lambda x: x is not None, storage))

    # combine dicts
    storage_dict = {k: v for x in storage for k, v in x.items()}

    # write data
    for k, v in storage_dict.items():
        data_file[k] = v

    # clean up
    data_file.close()

def process_dicom_series(target_size,input_tuple):
    """:param dir:
        target_size:
            the tuple specifying resolution (Y, X)
        input_tuple:
            [0]:
                the input path specifying a series  filled with dicom files
            [1]:
                the input path specifying a series  filled with contours files
            [2]:
                the image study key
    :return:
        the dictionary containing the labeled dicom files with a label
    """
    # get tuple values
    dicom_path = input_tuple[0]
    contour_path = input_tuple[1]
    image_study_id = input_tuple[2]


    # read in dicom files and convert to image
    img = parse_dicom_file(dicom_path)['pixel_data']

    # read contours and convert them into masks
    width,height = np.array(img).shape
    mask = poly_to_mask(parse_contour_file(contour_path), width, height)

    # move to temp dictionary
    temp_storage = {}
    temp_storage[image_study_id + "/label/"] = mask

    # add images
    temp_storage[image_study_id + "/image/"] = img

    return temp_storage

# save and process the HDf5
process_hdf5(image_path_lst,key_lst,label_path_lst, settings_dict)
