% https://www.mathworks.com/help/deeplearning/ug/transfer-learning-using-pretrained-network.html

net = googlenet;
unzip('C:\Users\bert\Downloads\Holes.zip', 'C:\Users\bert\Downloads\Holes');
imds = imageDatastore('C:\Users\bert\Downloads\Holes\Holes', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

inputSize = net.Layers(1).InputSize;

% To retrain a pretrained network to classify new images, replace these two layers with new layers adapted to the new data set.
% Extract the layer graph from the trained network.
lgraph = layerGraph(net); 
% Replace the fully connected layer with a new fully connected layer that has number of outputs equal to the number of classes. 
% To make learning faster in the new layers than in the transferred layers, increase the WeightLearnRateFactor and BiasLearnRateFactor values of the fully connected layer.

numClasses = numel(categories(imdsTrain.Labels)); % output is the number of classes



newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
    
lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

% TRAIN NETWORK
% The network requires input images of size 224-by-224-by-3, but the images in the image datastores have different sizes.
% Use an augmented image datastore to automatically resize the training images.
% Specify additional augmentation operations to perform on the training images: randomly flip the training images along the vertical axis, and randomly translate them up to 30 pixels horizontally and vertically.
% Data augmentation helps prevent the network from overfitting and memorizing the exact details of the training images.

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% To automatically resize the validation images without performing further data augmentation, 
% use an augmented image datastore without specifying any additional preprocessing operations.
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Specify the training options. For transfer learning, keep the features from the early layers of the pretrained network (the transferred layer weights).
% To slow down learning in the transferred layers, set the initial learning rate to a small value.
% In the previous step, you increased the learning rate factors for the fully connected layer to speed up learning in the new final layers.
% This combination of learning rate settings results in fast learning only in the new layers and slower learning in the other layers.
% When performing transfer learning, you do not need to train for as many epochs.
% An epoch is a full training cycle on the entire training data set.
% Specify the mini-batch size and validation data.
% The software validates the network every ValidationFrequency iterations during training.
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');


% Train the network consisting of the transferred and new layers. By default, trainNetwork uses a GPU if one is available.
% This requires Parallel Computing Toolbox™ and a supported GPU device. For information on supported devices, see GPU Computing Requirements (Parallel Computing Toolbox).
% Otherwise, it uses a CPU. You can also specify the execution environment by using the 'ExecutionEnvironment' name-value pair argument of trainingOptions.
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

% CLASSIFY VALIDATION IMAGES
[YPred,scores] = classify(netTransfer,augimdsValidation);


% Display 10 sample validation images with their predicted labels.
idx = randperm(numel(imdsValidation.Files),25);
figure
for i = 1:25
    subplot(5,5,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

% Calculate the classification accuracy on the validation set.
% Accuracy is the fraction of labels that the network predicts correctly.
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

% For tips on improving classification accuracy, see Deep Learning Tips and Tricks.








