function [C, sigma] = dataset3Params(X, y, Xval, yval)

C = 1;
sigma = 0.3;

% ----- %

values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 25, 50, 100];

results = [];

l = length(values);

for c=1:l,
    for sigma=1:l,      
      testC = values(c);
      testSigma = values(sigma);
      
      model= svmTrain(X, y, testC, @(x1, x2) gaussianKernel(x1, x2, testSigma)); 
      predictions = svmPredict(model, Xval);
      
      testError = mean(double(predictions ~= yval));
      
      results = [results; testC, testSigma, testError];
    end
end

% ----- %

[minError, minIndex] = min(results(:,3));

C = results(minIndex,1);
sigma = results(minIndex,2);

end
