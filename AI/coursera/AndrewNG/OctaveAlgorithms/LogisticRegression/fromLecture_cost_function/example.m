% NOTE: Theta has to be R^d, where d>=2. == Theta has to be at least 2-dimensional vector.
%       Ths means, you can't optimize one-dimensional function.

% Setup parameters of optimization algo.
options = optimset('GradObj', 'on', 'maxIter', 100);
initialTheta = zeros(2, 1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);

optTheta, functionVal, exitFlag

% Results are:
% optTheta = [5.000; 5.000]  % Correct
% functionVal = 1.5777e-030
% exitFlag = 1  % Convergence status
