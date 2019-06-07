# Test_PipeLine_cardiacMR

I initially defined a few functions to be able 1) to get the first sub directories, 2) to generate the slice number (aka the file name for the .dcm files that are match with the name of the contours, 3) to generate two lists containing the paths for corresponding .dcm files and contour.txt files using the data and lookup table (i.e., link.csv). 

Also wrote a function (func = save_overlay_images()) to find, convert, and overlay the masks on top of the images for quality check. Saved the overlay images as .png format in 'output' folder. 

![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/overlay_SC-HF-I-1_68.png)


This function also returns three lists containing paths to the images, labels, and patient ids in order to use in the second phase for data training generator. 

    The functions are in src.utility


Using methods above allow us to visually inspect the contours and images that are matched. However, checking the slices before and after the designated slice might help. In this data for each contour we had related .dcm files but in larger scope it might not be the case. We might need to return missing images or most likely missing contours in future steps. A simple function to return missing data will be necessary for larger datasets. 

# store the data in HDF5

Although this is a small dataset and we potentially could have gotten away with saving the data as numpy up front but I usually store the data in a dictionary or hdf5. 

    see hdf5_write.py 

Here given the directories to images, contours, and lookup table, we create and save a hdf5 file with patient id as key and image and label as 'items'. The hdf5 output from this level is saved in 'data' folder.

depends on cpu and gpu power reading images from a single hdf5 might be a good idea. If cpu cache size is not adequette it might be problematic to use hdf5. In this case, memory-mapping the files with NumPy could be used. Also there is a major drawback of storing a lot of data in a single file, which is what HDF5 is designed for. But as long as there is a clear path to create such a file and the raw data still exist we might be able to retrieve it in case on corruption. 

# data generator

Then we need to batch the data to feed the neural network. I created a class for 'DataGenerator'. within the class defined a

    func = train_generator
    
to recieve the data from hdf5 file and using the patient ids and batch size yeild the image and label numpy files. At the end of each epoch the patient ids will be suffled for randomizing the batch contains. 

Then created a simple vanilla 2D Unet to make sure that it is compatible to my keras fit_generator. 

To verify the batch generation I used some steps of logging and created some debug and info level logging. I returned whether shuffling occured, then printed out the patient ids that are in the batches and also I asked it to let me know when the object is being made. Other levels of logging in a more organized manner would be necessary in larger datasets. 

Alternatively, I could have shuffle the name of the .dcm files inside the generator and read, convert contour to binary mask,and convert them all to numpy whithin the data generator without using the hdf5. Then I save a more clear logging paradiam to  store the path/name of the images that are used for training instead of saving the entire data and reading them through hdf5. In larger dataset with better processing power, we might need to do such a method instead.  

At the end, I trained this Unet with a couple epochs with 12 steps per epoch wich is 96(number of 2D iamges)/8 (batch size) and I checked the logging files (in 'log' folder) to make sure the shuffling and patient ids are correctly insterted. 

![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/Picture1.png)

# Phase 2
# part 1 

The visualized function was a little bit changed and now we have all the overlay of the contours on the image to be
generalizable to i and o contours. Also, there was a bug in the function that I needed to fix.
During the quality check I realized that the o-contours for patient SCD0000501 seem to have a shift and they
are out of place. The name of the contours and images were checked and they were matched. i-contours for this
patient were drawn correctly too. I beleive there was shift in the o-contour masks.
I tried some fliping, transposing, and rotating the masks vs the image but didn't see that much of a
consistent pattern! For this specific task I would just put them aside and exclude from further analysis at this stage!

updated save_overlay_images is available now. 

making sure that the outer contours are aligned with myocardium. 
![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/Picture_2.png)

In src.utility I added  func mask_gen in order to create outer mask and its corresponding inner and ring (myocardium) mask. 

# packages 

pandas,
numpy,
hdf5,
PIL,
pydicom or dicom ,
math,
logging,
