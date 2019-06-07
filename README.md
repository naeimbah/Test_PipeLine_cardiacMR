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

In src.utility I added 

    func = mask_gen 
    
in order to create outer mask and its corresponding inner and ring (myocardium) mask. 

Let's take a look at a ring:

![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/ring.png)

Now I vectorize the masks that are generated in mask_gen to analysis their histogram and for futher analysis:

    func = gen_vec

In src.utiliy!


# Histogram analysis

I generate histogram of the intensities for blood pool (inside i-contour) and myocardium (between i and o contour). 
Below are some histogram illustrations for 1) one single subject, 2) normalized histogram of the subject, and 3) our population (using matplotlib)

1) 
![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/hist_test_1.png)


2) 
![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/hist_norm_1.png)


3) 
![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/hist_1.png)

It seems that there are some valley in the histogram and A number of methods exist to find the valley between the two modal peaks in a histogram. I test a few here. 

There are many methods to do thresholding such as Simple Thresholding, Adaptive Thresholding, and Otsu’s Binarization. Here I explore Otsu’s Binarization to investigate whether I can extract the blood pool from the outer contour. 

# Otsu's Binarization 

I use opencv to conduct Otsu's binarization to predict the blood pool from the outer contour using some thresholding mechanism. 
Code for this purpose is available in otsu.py 

Then I overlay the predicted blood pool form otsu on top of outer contour to visualize the process:

![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/ostu_on_outer.png)

Also it seems like this prediction is a good fit on the blood pool:

![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/ostu_on_bloodpool.png)

# Evaluation

There are so many ways to evaluate whether a segmentation is working well or not. Here after eyeballing and quality check I decided to render some numbers and metrics to see how quantitatively a thresholding can distinguish these regions from each other.
Majorely I can look at:
- Overlap rate (or Jaccard similarity measure). It is defined as the ratio of the intersection of segmented lesion area A and ground truth lesion area (or manually segmented areas) B to the union of segmented lesion area A and ground truth area B.
- Under segmentation rate defines the proportion of the unsegmented lesion area U=|B-(A∩B)|.
- Over segmentation rate is defined as the ratio of the segmented non-lesion area V=|A-(A∩B)| and the ground truth area B.

Here I evaluated my segmentations using Jaccard Similarity Score (JSS) and Dice Coefficient and the code is avaiable in eval.py using sklearn 

It shows that for 37 contours both JSS and Dice had a reasonable scores. Although for a couple of cases with small blood pool Dice coefficient is not great. see below:

![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/compare_thr.png)

# Alternative approach -- morphology of the image 

    # Canny Edge Detection
There are some steps for this edge detection method. 
hypothesis: I can detect the edges inside the outer heart mask that include the outer and inner contours and by removing the outer contour I might get a good contour of the blood pool. 
 
 There are a few pre processing steps towards this method:
 
 First, I need to denoise the image using a guassian filter.
 
 Second, get the gradients of the image, see below:
 
![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/G.png)

Third, removing the non max pixels. Ideally, the final image should have thin edges. Thus, I must perform non-maximum suppression to thin out the edges. see below:

![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/supressed.png)
 
 Forth, now I need to conduct a through thresholding method such as double thresholding. 
 The double threshold step aims at identifying 3 kinds of pixels: strong, weak, and non-relevant:

- Strong pixels are pixels that have an intensity so high that we are sure they contribute to the final edge.
- Weak pixels are pixels that have an intensity value that is not enough to be considered as strong ones, but yet not   small enough to be considered as non-relevant for the edge detection.
- Other pixels are considered as non-relevant for the edge.


![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/threshold.png)

Fifth, Edge Tracking by Hysteresis:
Based on the threshold results, the hysteresis consists of transforming weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one.

![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/Hysteresis.png)

Well, it seems like Hystersis didn't help that much. 

Now I conduct a two step island removal to first get rid of islands inside blood pool, then removing the outer contour as much as possible. Then using erode and dilate I fill out the remaining edges to create a mask that represents blood pool. 


![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/morph_close.png)
![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/morph_7.png)

- Evaluation of this method is done using the same metrics as before and seems like my thresholding method worked slightly better. However, this morphological method was very rudimental and it could be more work done on how to greate the structure (SE) and other hyperparameters in the method. 


![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/compare_all_JSS.png)
![alt text](https://github.com/naeimbah/Test_PipeLine_cardiacMR/blob/master/output/compare_all_Dice.png)

code for this section is avaialbe at src.morph.py using openCV. 



# Deep Learning vs Heuristic approaches 
There are some deep learning options to create the contours on the blood pool from 2D U-net to 2D-GANS. Using some interpolation to get more contours for each patietns would help to increase the sample size. Also, training a model to map the outer contour to a inner contour could be insteresting. 

However, any deep learning method need quit big sample size as opposed to conventional methods. But on the other hand, intensity thresholding or morphological approaches that were described above could be very subject dependent. For instance, I can imaging if there is a scar in myocardium then the morphology approach could easily fail or even the thresholding. Or if there is an artifact inside the blood pool it could be problematic for threshodling method. Segmenting top and bottom of the heart would be challenging due to small size of the interested organs and mapping the outer edge to the inner edge might be a good approach to tackle that problem. 

Given that, if I can get my hands on a bigger sample size or augment a big data set using interpolations, I would rather a deep learning approach and I would use heuristic methods more in pre and post processing to polish the data unless I design a more sophesticated conventional method. 


# packages 

pandas,
numpy,
hdf5,
PIL,
pydicom or dicom ,
math,
logging,
cv2, 
sklearn, 
scipy

