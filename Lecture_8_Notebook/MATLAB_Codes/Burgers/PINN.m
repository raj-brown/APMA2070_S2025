clc;
tic;
numBoundaryConditionPoints = [25 25];
x0BC1 = -1*ones(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));
t0BC1 = linspace(0,1,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,1,numBoundaryConditionPoints(2));
u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));
numInitialConditionPoints  = 50;
x0IC = linspace(-1,1,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);
u0IC = -sin(pi*x0IC);
X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];
numInternalCollocationPoints = 10000;
pointSet = sobolset(2);
points = net(pointSet,numInternalCollocationPoints);
dataX = 2*points(:,1)-1;
dataT = points(:,2);
ds = arrayDatastore([dataX dataT]);

parameters = struct;
numLayers = 9;
numNeurons = 20;
sz = [numNeurons 2];
parameters.fc1.Weights = initializeHe(sz,2,"single");
parameters.fc1.Bias = initializeZeros([numNeurons 1],"single");

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn,"single");
    parameters.(name).Bias = initializeZeros([numNeurons 1],"single");
end

sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers).Weights = initializeHe(sz,numIn, "single");
parameters.("fc" + numLayers).Bias = initializeZeros([1 1], "single");

parameters;

numEpochs = 20000 ; %3000;
miniBatchSize = 10000;
executionEnvironment = "auto";

initialLearnRate = 0.01;
decayRate = 0.005;

mbq = minibatchqueue(ds, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFormat="BC", ...
    OutputEnvironment=executionEnvironment);

X0 = dlarray(X0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(U0);

if (executionEnvironment == "auto" && canUseGPU) || ...
        (executionEnvironment == "gpu")
    X0 = gpuArray(X0);
    T0 = gpuArray(T0);
    U0 = gpuArray(U0);
end

averageGrad = [];
averageSqGrad = [];
lossHist = zeros(numEpochs,1);

accfun = dlaccelerate(@modelLoss);

% figure
% C = colororder;
% lineLoss = animatedline(Color=C(2,:));
% ylim([0 inf])
% xlabel("Iteration")
% ylabel("Loss")
% grid on
start = tic;
iteration = 0;
for epoch = 1:numEpochs
    reset(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        XT = next(mbq);
        X = XT(1,:);
        T = XT(2,:);
        [loss,gradients] = dlfeval(accfun, parameters,X,T,X0,T0,U0);
        learningRate = initialLearnRate / (1+decayRate*iteration);
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,...
            gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end
    loss = double(gather(extractdata(loss)));
    lossHist(epoch) = loss;
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    fprintf('On Epoch: %d, and Loss %f\n', epoch, loss)   
end
toc;


tTest = [0.25 0.5 0.75 1];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);

figure

for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);

    % Make predictions.
    XTest = dlarray(XTest,"CB");
    TTest = dlarray(TTest,"CB");
    UPred = model(parameters,XTest,TTest);

    % Calculate true values.
    UTest = solveBurgers(extractdata(XTest),t,0.01/pi);

    % Calculate error.
    err = norm(extractdata(UPred) - UTest) / norm(UTest);

    % Plot predictions.
    subplot(2,2,i)
    plot(extractdata(XTest),extractdata(UPred),"--r",LineWidth=2);
    ylim([-1.1, 1.1])

    % Plot true values.
    hold on
    plot(extractdata(XTest), UTest, "-b",LineWidth=2.2)
    hold off

    title("t = " + t + ", Error = " + gather(err));
end

subplot(2,2,2)
legend("Predicted","True")
ep = 1:numEpochs;
close
semilogy(ep, lossHist, "-k", linewidth=2.0)
xlabel("Iteration", fontsize=18)
ylabel("Loss", fontsize=18)
ax = gca;
ax.FontSize=20;