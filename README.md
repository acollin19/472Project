# Comp 472 Project Assignment, Part 1

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
1. Run training.py which calls preprocessing.py to obtain train_loader and test_loader, then trains the data using the layers defined in the cnn class (40 epochs).
   - Saves model into saved_model file

### Application  
2. Once the saved model is obtained...
3. Run confmatrix.py which uses the model (modelB) obtained from training.py to obtain all parameters and the confusion matrix.  






