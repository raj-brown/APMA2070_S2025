function out = dotProduct(input1, input2)
    out = sum(input1 .* input2, 2);  % Element-wise multiplication and summing over the results
end
