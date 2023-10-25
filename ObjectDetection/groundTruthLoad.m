% https://www.mathworks.com/help/vision/ref/objectdetectortrainingdata.html

%%%%%%%%%%%%%%%%%%%%%%
% If we want to add auto opening Image labeler for their own folder of
% images
% Add the folder containing images to the MATLAB path
% imageDir = fullfile(matlabroot, 'AerialPhotos');
% addpath(imageDir);
%%%%%%%%%%%%%%%%%%%%%%

% Adjust or load folder for image training
% Adjust threshold for annotation confidence scores

% Load ground truth data object(exported from image labeler)
load('gTruthTesting.mat')

% Use only the hole ground truth label (incase we add more)
holeGTruth = selectLabelsByName(gTruth, 'Hole');

% Make sure image labels are of type 'rectangle'
holeGTruth.LabelDefinitions

% Create a folder named TrainingData in our current folder to store
% training images and add to the path
if isfolder(fullfile('TrainingData'))
    cd TrainingData
    addpath('TrainingData');
else
    mkdir TrainingData
end

% Extract a subset of the ground truth data set (Sampling factor is how
% often from ground truth data an image is sampled)
trainingData = objectDetectorTrainingData(holeGTruth, 'SamplingFactor',3,'WriteLocation', 'TrainingData', 'Verbose', true);

% Train the detector
detector = trainACFObjectDetector(trainingData,'NumStages',5);

%%%%%%%%%%% ADDED TO UPLOAD IMAGE IN APP
% Test the ACF-Based Detector on a sample image
I = imread('18.png');
[bboxes, scores] = detect(detector,I);
% Display the detected test image
annotations = strings(length(scores),1);
for i=1:length(scores)
    annotations{i} = ['Confidence: ' num2str(scores(i),'%0.2f') '%'];
end
detectedImage = insertObjectAnnotation(I,'rectangle',bboxes,annotations);
figure
title("Detected Holes Test Image")
imshow(detectedImage)
%%%%%%%%%%%%%%%%%%%

% Change directory back up to load script
cd ..
% Save the detector to a MAT file
save('Detector.mat', 'detector');





