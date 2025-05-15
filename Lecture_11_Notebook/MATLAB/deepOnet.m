% Define input dimensions
inputDimBranch = 100;  % Number of points for u(x)
inputDimTrunk = 1;     % A single point for y
hiddenUnits = 64;

% Branch network (for function u(x))
branchLayers = [
    featureInputLayer(inputDimBranch, "Name", "branch_input")
    fullyConnectedLayer(hiddenUnits, "Name", "branch_fc1")
    reluLayer("Name", "branch_relu1")
    fullyConnectedLayer(hiddenUnits, "Name", "branch_fc2")
    ];

% Trunk network (for point y)
trunkLayers = [
    featureInputLayer(inputDimTrunk, "Name", "trunk_input")
    fullyConnectedLayer(hiddenUnits, "Name", "trunk_fc1")
    reluLayer("Name", "trunk_relu1")
    fullyConnectedLayer(hiddenUnits, "Name", "trunk_fc2")
    ];

% Dot product layer (combines the outputs of the Branch and Trunk networks)
dotProductLayer = functionLayer(@(inputs) dotProduct(inputs{1}, inputs{2}), "Name", "dot_product");

% Final layers
finalLayers = [
    dotProductLayer
    fullyConnectedLayer(1, "Name", "fc_out")
    regressionLayer("Name", "regression_output")
    ];
% Create the layer graph and add all layers
lgraph = layerGraph();

% Add the Branch and Trunk networks
lgraph = addLayers(lgraph, branchLayers);
lgraph = addLayers(lgraph, trunkLayers);
lgraph = addLayers(lgraph, finalLayers);

% Connect Branch network output to the dot product layer input 1
lgraph = connectLayers(lgraph, "branch_fc2", "dot_product/in1");

% Connect Trunk network output to the dot product layer input 2
lgraph = connectLayers(lgraph, "trunk_fc2", "dot_product/in2");


% Parameters
nSamples = 1000;    % Number of training samples
inputDimBranch = 100; % Number of discretization points for u(x)

% Generate x values and corresponding u(x)
x = linspace(0, 1, inputDimBranch);
u_data = sin(pi * x);            % Example: u(x) = sin(pi*x)
U = repmat(u_data, nSamples, 1); % Repeat u(x) for each training sample

% Generate y values (point inputs)
y = rand(nSamples, 1);  % Random values for y (between 0 and 1)

% Compute target values (output)
targets = cos(pi * y);  % Example: Target = cos(pi * y)

% Create individual datastores
dsU = arrayDatastore(U, 'IterationDimension', 1);        % Branch input
dsY = arrayDatastore(y, 'IterationDimension', 1);        % Trunk input
dsT = arrayDatastore(targets, 'IterationDimension', 1);  % Output

% Combine datastores
dsCombined = combine(dsU, dsY, dsT);

% Transform datastore into required format
dsFinal = transform(dsCombined, @(data) {data{1}, data{2}, data{3}});


options = trainingOptions("adam", ...
    "MaxEpochs", 300, ...        % Maximum number of training epochs
    "MiniBatchSize", 32, ...     % Mini-batch size
    "InitialLearnRate", 1e-3, ...% Initial learning rate
    "Shuffle", "every-epoch", ...% Shuffle the data every epoch
    "Verbose", false, ...        % Display progress during training
    "Plots", "training-progress"); % Plot training progress

% Train the network
net = trainNetwork(dsFinal, lgraph, options);


% Test data (same as training data)
yt = linspace(0, 1, 100)';          % New test values for y
Ut = repmat(u_data, length(yt), 1);  % Repeat u(x) for each y value

% Predict with the trained network
pred = predict(net, {Ut, yt});

% Plot the results
plot(yt, cos(pi*yt), 'k-', yt, pred, 'r--')
legend("True", "Predicted")
title("DeepONet Prediction (Dot Product)")
