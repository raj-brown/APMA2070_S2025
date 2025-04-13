clc; clear; close all;

% Define network parameters
branchInputSize = 100;   % Number of sensors for function input
trunkInputSize = 1;      % Single evaluation point
hiddenLayerSize = 50;    % Number of neurons in hidden layers
outputSize = 1;          % Output scalar value

% Define branch network (fully connected layers)
branchLayers = [
    featureInputLayer(branchInputSize, 'Name', 'branchInput')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'branchFC1')
    reluLayer('Name', 'branchReLU1')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'branchFC2')
    reluLayer('Name', 'branchReLU2')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'branchFC3')
];

% Define trunk network (fully connected layers)
trunkLayers = [
    featureInputLayer(trunkInputSize, 'Name', 'trunkInput')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'trunkFC1')
    reluLayer('Name', 'trunkReLU1')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'trunkFC2')
    reluLayer('Name', 'trunkReLU2')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'trunkFC3')
];

% Define the final output network (dot product + output layer)
outputLayers = [
    additionLayer(2, 'Name', 'merge') % Merge branch and trunk outputs
    fullyConnectedLayer(outputSize, 'Name', 'output')
];

% Combine all layers into a deep learning network
lgraph = layerGraph(branchLayers);
lgraph = addLayers(lgraph, trunkLayers);
lgraph = addLayers(lgraph, outputLayers);

% Connect branch and trunk outputs to the merge layer
lgraph = connectLayers(lgraph, 'branchFC3', 'merge/in1');
lgraph = connectLayers(lgraph, 'trunkFC3', 'merge/in2');

% Visualize the network
figure;
plot(lgraph);
title('DeepONet Architecture');

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Generate synthetic training data (example)
numSamples = 1000;
X_branch = rand(numSamples, branchInputSize);
X_trunk = rand(numSamples, trunkInputSize);
Y = sum(X_branch, 2) .* X_trunk; % Example function output

% Convert data to datastores
dsTrain = arrayDatastore({X_branch, X_trunk}, 'IterationDimension', 1);
dsResponse = arrayDatastore(Y, 'IterationDimension', 1);
trainData = combine(dsTrain, dsResponse);

% Train the DeepONet model
net = trainNetwork(trainData, lgraph, options);

% Test the model on new data
X_test_branch = rand(10, branchInputSize);
X_test_trunk = rand(10, trunkInputSize);
predictions = predict(net, {X_test_branch, X_test_trunk});

disp('Predicted outputs:');
disp(predictions);
