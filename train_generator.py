# Now I will import those hdf5 information into the data generator
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
import keras
import math
import logging
from src.sample_Unet_2D import get_unet

settings_dict = {
        "target_size": (256,256),
        "data_path": "/home/nabahrami/Documents/Naeim/PipeLine/data/data.hdf5"
                 }

# read the data from the hdf5
data_hdf5 = h5py.File(settings_dict["data_path"], "r")

# extract the list of ids (i.e., 2D images with their masks)
list_IDs = list(data_hdf5.keys())

# other parameters
batch_size = 8
N_epochs = 2

# create some supervision
logging.basicConfig(filename='example.log',level=logging.INFO)


# class of DataGenerator
class DataGenerator:

    # init function definition and setting the shuffle to true
    def __init__(self, list_IDs,data_hdf5, batch_size=8,
                  shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data_hdf5 = data_hdf5
        self.shuffle = shuffle

        # let me know if you are making an object
        logging.info("making object")

    # We can use the function below to shuffle the ids after the end of each epoch
    # def on_epoch_end(self,list_IDs):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(list_IDs))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    # a function for batch generating
    def train_generator(self):
        start_idx = 0
        end_idx = self.batch_size

        curr_id_lst = self.list_IDs

        # if shuffle needed then after each epoch randomize the image ids
        if self.shuffle:
            #let me know if you are shuffling
            logging.info("shuffling")
            np.random.shuffle(curr_id_lst)

        # a while true loop with yeilding the image and labels as npy arrays fit for a sample u-net
        while True:
            if end_idx > len(curr_id_lst):
                start_idx = 0
                end_idx = self.batch_size

            batch_img = []
            batch_label = []
            for id in curr_id_lst[start_idx:end_idx]:

                # let me know with the ids in the batch
                logging.info("adding key key: {}".format(id))
                batch_img.append(list(self.data_hdf5[id]['image']))
                batch_label.append(list(self.data_hdf5[id]['label']))

            start_idx = start_idx + self.batch_size
            end_idx = end_idx + self.batch_size
            yield np.expand_dims(np.array(batch_img),3), np.expand_dims(np.array(batch_label),3)


my_gen_obj = DataGenerator(list_IDs,data_hdf5, batch_size=8,
              shuffle=True)

# define the steps per epoch as length of the id list divid by the batch size
steps_per_epoch = math.ceil(len(list_IDs)/batch_size)

# test the data generator on a sample u-net
train_model = get_unet()
train_model.fit_generator(
    generator=my_gen_obj.train_generator(),
    epochs=N_epochs,
    steps_per_epoch=steps_per_epoch)
