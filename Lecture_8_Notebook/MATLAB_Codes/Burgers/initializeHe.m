function weights = initializeHe(sz,numIn, type)
weights = randn(sz,type) * sqrt(2/numIn);
weights = dlarray(weights);
end