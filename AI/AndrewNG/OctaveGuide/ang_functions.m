my_function(5);  % Calls my_function function from separate my_function.m file

% Octave search path:
% addpath('C:\Users\max\...');

[sq, cube] = my_func_2(2);

% Goal: define a function to compute the cost function J(Theta)
X = [
  1 1;
  1 2;
  1 3;
];
y = [
  1;
  2;
  3;
];
theta = [0;1];
j = costFunctionJ(X, y, theta)
theta = [0;0];
j = costFunctionJ(X, y, theta)  % 2.333 == (1^2+2^2 + 3^2) / (2*m)


