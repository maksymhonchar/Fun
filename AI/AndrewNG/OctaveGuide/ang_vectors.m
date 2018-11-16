A = [1 2; 3 4; 5 6;];
B = [
  1 2;
  3 4;
  5 6;
];
m_13 = [1 2 3];
v_31 = [
  1;
  2;
  3;
];

v_range= 1:0.1:2;  % 1x11 matrix
v_range_2 = 1:6;  % 1x6 matrix

ones(2,3);
ones(3);

C = 2 * ones(2,3);  % same as C=[2 2 2; 2 2 2]

w = ones(1,3);
w_z = zeros(1,3);

w = rand(1,3);
w = rand(3,3);

w = randn(1,3);  % Gaussian distribution

w = -6 + sqrt(1) * randn(1,10000);  % vector with 10000 elements
% hist(w);  % Create a histogram
% hist(w, 50);

w = magic(4);

eye(4);  % Identity matrix

% help eye % documentation for function
% help help

