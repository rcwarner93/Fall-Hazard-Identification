# Fall-Hazard-Identification (MATLAB R2023a)
Machine Learning using ACF model to identify holes in construction sites using drone 2-D aerial imaging

MATLAB Tool Boxes Required:
- Computer Vision (Image Label App for Ground Truth)
- Deep Learning 

Installation: https://matlab.mathworks.com/
- Install MATLAB R2023a
- Open the Fall-Hazard-Identification folder in the workspace

Loading the application: 
    - Navigate to the ObjectDetection folder and open the holeDetectionApp.mlapp in MATLAB and press run (Play button at the top)

Modifying the current ground truth labels: 
    1 - Open the Image Labeler app under the APPS tab at the top
    2 - Click Open Project at the top left
    3 - Navigate to the LabelingProject folder and open the LabelingProject.prj file

    - If the image path needs to be resolved for the data training images: (0 of XX images found. Location of images..)
        - Click on new location in the right side, browse to the ObjectDetection->TrainingData folder and press resolve
        - From here you can edit the labels or remove them as needed

    - To add new images to the ground truth:
        - Place them in the TrainingData folder
        - Import them in the Image Labeler app
        - From there you should be able to add Hole ROI labels to the imported images

    - When finished, press the Export button in the top menu bar, navigate to the object detection folder, and select the gTruth.mat file to save the file
    - NOTE:
        - The above export will overwrite the old ground truth file
        - Don't forget to save the Labeling project when exiting!
    - Retrain the model with the added images by running the ACFdetectortraining.m file
        - This is done by entering ACFdetectortraining in the command prompt window
        - This script will
            -Retrain the model and save to a Detector.mat file with the new model
            -Show a test image display of annotation boxes for a sanity check
            -Display a figure with two graphs showing average precision (PR-Curve) and Log Average Missrate
        
    - Rerun the application and it will load the new Detector.mat file at startup

YOLOannotationGen
    - Generates YOLO annotation .txt files for each image labeled in MATLAB's Image Labeler App in a new folder called YOLOAnnotations
    - These .txt bounding box list annotation files can be used for other coding languages or to train a YOLO model

YOLOdetectorTraining
    - This is a test script for implementing a different algorithm model for object detection using YOLO
    - We could not get it to work fully due to the differing sizes of images in the database
    - Resizing may lead to issues with the generated YOLO annotations
    - It was kept as part of our stretch goal for using different algorithms to solve the problem
    - Can be used as a building block for later implementation
