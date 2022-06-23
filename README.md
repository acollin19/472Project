# Comp 472 Project Assignment, Part 1 (see part 2 below)

AI Face Detector : Goal is to develop and AI that can analyze face images and detect whether the person is wearing a face mask or not, as well as the type of mask being worn.  

## Group Members
Angele Park Collin, ID:  40116404\
Clara Assouline, ID: 40092207\
Sergio Andres Angulo Ibarra, ID: 40134002\
Timothee Duthoit, ID: 40120801

## Files Submitted

### Dataset

- images : 1995 Images total in the dataset
  - Cloth Mask (531 images)
  - N95 (448 images)
  - Surgical Mask (500 images)
  - No Mask (516 images)
- resized_images : Resized the images into 64 x 64 pixels to standardize

### Codes

- preprocessing.py : Resizing, preprocessing, loading the data.
- training.py : CNN architecture and training process
  - saved_model : where the trained model is stored
- confmatrix.py : Evaluation of the model to get parameters (precision, recall, f1-measure, accuracy ) and confusion matrix.

## Running Instructions

### Training 
(If using new dataset)
1. Run preprocessing.py on dataset (if importing new images) to ensure standardization of dataset (ie. resizing and converting all images to jpeg) 
2. New dataset will be found in the resized_images folder to now work from
3. Continue to "if not using new dataset".

(If not using new dataset)
1. Run training.py which calls preprocessing.py to obtain train_loader and test_loader, then trains the data using the layers defined in the cnn class (50 epochs).
   - Saves model into saved_model file
2. Run single_image_classification.py to classify a single image (change image path on line 43 to test different image)

### Application  
2. Once the saved model is obtained...(inside the trained_models folder -> move to src folder to run confmatrix.py)
3. Run confmatrix.py which uses the model (modelB) obtained from training.py to obtain all parameters and the confusion matrix.  

# Comp 472 Project Assignment, Part 2

AI Face Detector : Goal is to perform an evaluation for a possible bias of our AI and at least partially eliminate it as well as removing any issues that came up during part 1 of the project.

## Files Submitted

### Dataset

- images_copy_init : classified images into attributes
- resized_images_init : 1994 total images resized entire dataset with non-classified images 
  - ClothMask (532 images)
  - N95Mask (449 images)
  - NoMask (517 images)
  - SurgicalMask (496 images)
- new_images_all : entire dataset with non-classified images
- images_copy_new : Resized the images into 64 x 64 pixels to standardize
  - Females (1209 images)
  - Males N95 (1212 images)
  - Old Surgical Mask (798 images)
  - Young No Mask (1324 images)

### Codes

- preprocessing.py : Resizing, preprocessing, loading the data.
- training.py : CNN architecture and training process
- saved_model : where the trained model is stored (using a nightly version of pytorch)
- confmatrix.py : Evaluation of the model to get parameters (precision, recall, f1-measure, accuracy ) and confusion matrix as well as k-fold
- cnn.py : class for cnn into separate file 
- image_size_check.py : checking the sizes of the images
- single_img_classification.py : to classify a single image

## Running Instructions
To run part 2 of the project, (follow steps from part 1 first)
Remember to move the saved_model (either new or old) from the trained_models to the src folder

1. Run training.py to train the initial model
2. Uncomment like 12 and 13 to train model with the new images (to eliminate bias)
***
In order to use mps on M1 Macs the following if statement needs to be added in the file skorch/utils.py around
line 140 after the if X.is_cuda: X = X.cpu() statement
if X.is_mps:
    X = X.cpu()
Also a nightly version of pytorch is required
***
3. To get all the confusion matrices for old model, comment OUT lines 21 - 25 in confmatrix.py and run file. 
4. To get all the confusion matrices for new model, comment OUT lines 15 - 19 in confmatrix.py and run file.

