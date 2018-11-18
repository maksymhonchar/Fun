A = [
  1 2;
  3 4;
  5 6;
];
B = [
  11 12;
  13 14;
  15 16;
];
C = [
  1 1;
  2 2;
];

mult_AC = A * C;
element_wise_mult_AB = A .* B;  % Aij*Bij
element_wise_squaring_A = A .^ 2;  % Aij^2

v = [1; 2; 3];
1 ./v;  % [1/1; 1/2; 1/3]
1 ./ A;  % [1/ 1/; 1/ 1/; 1/ 1/]

log(v);  % element_wise logarithm ln

abs(v);  % element_wise absolute value of v
abs( [-1 -2 -3] );

% increment each element of v by 1:
v + ones(length(v), 1);
v + 1;

A;
A';  % Transposing matrix
(A')';

a = [1 15 2 0.5];
val = max(a);  % maximum value of a: 15
[val, ind] = max(a);  % maximum value and its index

A;
a < 3;  % applying to matrix/vector, this is ELEMENT_WISE comparison [1 0 1 1]
A < 3;

find(a < 3);  % returns indexes of values less than 3
find(A < 3);  % returns indexes of ROWS less than 3

% magic is not usable for ML!
A = magic(3);  % return 'magic square' -> all rows, cols and diags sum up same value

[r, c] = find(A >= 7);  % returns separated pairs (value from r, value from c)

a;
sum(a);  % sum of all values in a
prod(a);  % multiplication of all values in a
floor(a);  % round down
ceil(a);  % round up

rand(3);  % random 3x3 matrix

rand_A = rand(3);
rand_B = rand(3);
max(rand_A);  % max value from EACH column
max(rand_A, rand_B);  % for each cell -> get max value from rand_A and rand_B

A;
max(A, [], 1);  % max COLUMN WISE values
max(A, [], 2);  % max ROW WISE values

% To find max value from matrix:
max(max(A));
max(A(:));

A = magic(9);
sum(A, 1);  % sum of columns
sum(A, 2);  % sum of rows

% To sum diagonals:
main_diag = eye(9);
only_diag_values = A .* main_diag;  % let's try main diagonal
sum(sum(only_diag_values));

% opposite to main diagonal
opposite_main_diag = flipud(eye(9));
opposite_diag_values = A .* opposite_main_diag;
sum(sum(opposite_diag_values));

% Inverting matrix
A = magic(3)
inv_A = pinv(A);
inv_A * A  % eye(3)
A * inv_A  % eye(3)
