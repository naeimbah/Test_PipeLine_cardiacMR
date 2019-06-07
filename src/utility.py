
import os
import re
import pydicom
from pydicom.errors import InvalidDicomError
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# # directories
# dicom_dir = '/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/dicoms/'
# contour_dir = '/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/contourfiles/'
# # load csv in pandas dataframe format
# lookup_table = pd.read_csv('/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/link.csv')



def parse_contour_file(filename):
    """Parse the given contour filename

    :param filename: filepath to the contourfile to parse
    :return: list of tuples holding x, y coordinates of the contour
    """

    coords_lst = []

    with open(filename, 'r') as infile:
        for line in infile:
            coords = line.strip().split()

            x_coord = float(coords[0])
            y_coord = float(coords[1])
            coords_lst.append((x_coord, y_coord))

    return coords_lst


def parse_dicom_file(filename):
    """Parse the given DICOM filename

    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    try:
        dcm = pydicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        dcm_dict = {'pixel_data' : dcm_image}
        return dcm_dict
    except InvalidDicomError:
        return None


def poly_to_mask(polygon, width, height):
    """Convert polygon to mask

    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask

# Checking wether the images and masks are compatible
def get_immediate_subdirectories(dir):
    """Generate the immidiate subdirectories
    :param dir: path of the directory (type = str)
    :return: list of subdirectories
    """
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]


# writing a function to extract the slice number (contour name that is match with .dcm name)
def contour_name_2_contour_number(contour_name):
    """Generate the intiger number connecting the contour to the .dcm file (i.e., slice number)
    :param dir: name of the contour
    :return: number of the contour and .dcm file
    """
    contour_name_split_lst = contour_name.strip().split('-')
    return int(contour_name_split_lst[2])


def pat_lst_gen(dicom_dir,lookup_table):
    """Generate two lists: 1) list of names of folders with dicom files and 2) list of names of folders with corresponding contours
    :param dir: directory to dicom files
    """
    # get the list of patients with dicom files
    dicom_sub_dirs_lst = get_immediate_subdirectories(dicom_dir)

    # get the corresponding contour id using one line conditional processing
    contour_sub_dirs_lst = []
    for i in range(len(dicom_sub_dirs_lst)):
        contour_sub_dirs_lst.append(lookup_table[lookup_table['patient_id'] == dicom_sub_dirs_lst[i]]['original_id'].values[0])

    return dicom_sub_dirs_lst,contour_sub_dirs_lst


def save_ovelay_images(contour_dir, dicom_dir,lookup_table,contours):
    """save the images with overlay of the masks in png files
    :param dir: directories of contours and dicoms and list of directories with contour and dicom files
    :return: figure with overlay of mask on the image
                also it returns list of labels, dicom images, and pateint ids that can be used for data storage and data generator
    """

    image_id_lst = []
    dicom_path_lst = []
    label_path_lst = []

    # getting list of sub directories with dicom and contour files that are corresponded
    dicom_sub_dirs_lst,contour_sub_dirs_lst = pat_lst_gen(dicom_dir,lookup_table)

    for i in range(len(contour_sub_dirs_lst)):

        # getting the ith original_id contours
        # contour_dir = '/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/contourfiles/'
        contour_path = contour_dir + contour_sub_dirs_lst[i]+contours + '/'


        # getting the ith patient_id dicoms
        # dicom_dir = '/home/nabahrami/Documents/Naeim/PipeLine/Raw/final_data/dicoms/'
        dicom_path = dicom_dir + dicom_sub_dirs_lst[i]

        for _, _, fileList in os.walk(contour_path):
            for contour_name in fileList:

                # read individual contour
                coords_lst = parse_contour_file(os.path.join(contour_path , contour_name))
                label_path_lst.append(os.path.join(contour_path , contour_name))


                # finding the slice number or .dcm file
                slice_number = contour_name_2_contour_number(contour_name)

                # read individual dicom
                dicom_path_number = dicom_path + '/' + str(slice_number)+ '.dcm'
                dcm_dict = parse_dicom_file(dicom_path_number)
                dicom_path_lst.append(dicom_path_number)


                # converting contours into masks
                polygon = coords_lst
                width,height = dcm_dict['pixel_data'].shape
                mask = poly_to_mask(polygon, width, height)

                image_id_lst.append([dicom_sub_dirs_lst[i]+'_'+str(slice_number)])

                plt.imshow(dcm_dict['pixel_data'],cmap = 'gray')
                plt.imshow(mask,alpha = 0.2)
                plt.savefig('/home/nabahrami/Documents/Naeim/PipeLine/output/{}_overlay_{}_{}'.format(contours,contour_sub_dirs_lst[i],slice_number))
                plt.show()
                print(dicom_path)
                print(dicom_path_number)
                print(contour_name)

    return label_path_lst,dicom_path_lst,image_id_lst


# get list of labels, dicom images, and pateint ids

# label_path_lst,dicom_path_lst,image_id_lst = save_ovelay_iamges(contour_dir, dicom_dir,lookup_table)

def mask_gen(dicom_dir,contour_dir,dicom_lst, contour_lst):
        """Generate three sets of masks: 1) the i contours 2) corresponding o contours and 3) the ring that indicates myocardium
        :param dir: list of dicoms and their corresponding contours.
        """
        img = []
        o_contour_out = []
        i_contour_out = []
        diff = []
        for i in range(len(contour_lst)):

            # getting the outer contour path
            contour_path = contour_dir + contour_lst[i]+ '/o-contours/'

            # getting the dicom path
            dicom_path = dicom_dir + dicom_lst[i]

            for _, _, fileList in os.walk(contour_path):
                for contour_name in fileList:

                    # read the outer contour
                    coords_lst = parse_contour_file(os.path.join(contour_path , contour_name))

                    # finding the slice number or .dcm file
                    slice_number = contour_name_2_contour_number(contour_name)

                    # read individual dicom
                    dicom_path_number = dicom_path + '/' + str(slice_number)+ '.dcm'
                    dcm_dict = parse_dicom_file(dicom_path_number)

                    # converting contours into masks
                    polygon = coords_lst
                    width,height = dcm_dict['pixel_data'].shape
                    mask_o = poly_to_mask(polygon, width, height)

                    img.append(dcm_dict['pixel_data'])
                    o_contour_out.append(mask_o)

                    # find corresponding inner contour with the outer contour and create its mask
                    contour_path_i = contour_dir + contour_lst[i]+ '/i-contours/'
                    contour_name_i = re.sub(r'ocontour','icontour',contour_name)
                    # print(contour_name_i)
                    coords_lst_i = parse_contour_file(os.path.join(contour_path_i , contour_name_i))
                    # converting contours into masks
                    polygon_i = coords_lst_i
                    mask_i = poly_to_mask(polygon_i, width, height)
                    i_contour_out.append(mask_i)

                    # create the difference between outer and inner contour
                    # This indicates the ring or myocardium
                    diff.append([np.subtract(mask_o.astype(np.float32),mask_i.astype(np.float32))])

        return np.array(img),np.array(o_contour_out),np.array(i_contour_out), np.array(diff)[:,0,:,:]
