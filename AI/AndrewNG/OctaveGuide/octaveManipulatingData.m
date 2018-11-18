% Creating variables.
a = 3;
b = 'hi';
b_2 = "hi";
c = (3 >= 1);
A = [ 1 2; 3 4; 5 6 ];
B = [
  1, 2;
  3, 4;
  5, 6;
];
v = [1; 2; 3];
v_range = [1:0.1:2];  % [1;2] with step 0.1. 1x11 matrix.
v_range_2 = 1:6;  % Possible to create without brackets. 1x6 matrix.
a < 3;  % applying to matrix/vector, this is ELEMENT_WISE comparison.
A < 3;
A_23 = A(2,3)  % Indexing into the 2nd row 3rd column of matrix A.

% Matrices
A = eye(5);  % Identity matrix.
A = ones(2,3);
A = zeros(3,2);
A = rand(5,10);
A = randn(5,10);  % Gaussian distribution.
A = magic(10);  % Magic matrix: cols, rows, diags have the same sum.
C = 2 * ones(2,3);  % same as C=[2 2 2; 2 2 2]

% Some logic operators.
1==2;  % EQUALS
1~=2;  % NOT EQUALS
1&&0;  % AND
1||0;  % OR
xor(1,0);  % XOR

% Displaying data. Use disp, display or value call without semicolon.
a = pi;
display("displaying [a] variable in next line:");
disp(a);  % printing variables
% Use sprintf to make some c-like string formatting.
disp(sprintf('2 decimals: %0.2f', a));
disp(sprintf('6 decimals: %0.6f', pi));

% Calculations of different data types.
mult_AC = A * C;
% Note, that [.Operation] stands for elementwise operation
element_wise_mult_AB = A .* B;  % Aij*Bij
element_wise_squaring_A = A .^ 2;  % Aij^2
1 ./v;  % [1/; 1/; 1/]
1 ./ A;  % [1/ 1/; 1/ 1/; 1/ 1/]
log(v);  % element_wise logarithm ln.
abs( [-1 -2 -3] );  % element_wise absolute value of matrix.

% Increment each element of v by 1:
v + ones(length(v), 1);
v + 1;

% Transposing matrix
A;
A';
(A')';

% ":" means every element along that row/column.
second_row = A(2,:);  % Second row of A.
second_column = A(:,2);  % Second column of A.
first_ten_entries = data1(1:10);
A([1 3], :);  % Get everything from first and third row; Get all from columns.
A([1 3], 1);  % Get everything from 1st and 3rd row; Get data from 1 columns.
% Sophisticated assignment:
A(:, 2) = [
  10;
  11;
  12
];
% Put all elements of A matrix into a single vector
A(:);

% Append another column vector to right.
A = [A, [100; 101; 102]];
C = [A B];  % Concatenating of the matrices.
C = [A, B];  % same as C=[A B].
C = [A; B];  % Put the B at the bottom.

% Inverting matrix
A = magic(3);
inv_A = pinv(A);  % sudo inv
inv_A * A  % ~eye(3)
A * inv_A  % ~eye(3)

% Get the dimension of the matrix A where m=rows n=columns.
[m, n] = size(A)
dim_A = size(A)

% length() function.
v = [1 2 3 4];
longest_dimension_v = length(v);  % 4

% Maximum values of matrix
a = [1 15 2 0.5];
val = max(a);  % maximum value of a: 15
[val, ind] = max(a);  % maximum value and its index
max(rand_A, rand_B);  % for each cell -> get max value from rand_A and rand_B
max(A, [], 1);  % max COLUMN WISE values
max(A, [], 2);  % max ROW WISE values
% Find a single max value from matrix.
max(max(A));
max(A(:));

% Magic matrices.
A = magic(9);
sum(A, 1);  % sum of columns - same
sum(A, 2);  % sum of rows - same
% Sum diagonals: main diagonal - same.
main_diag = eye(9);
only_diag_values = A .* main_diag;
sum(sum(only_diag_values));
% Sum diagonals: opposite to main diagonal - same.
opposite_main_diag = flipud(eye(9));
opposite_diag_values = A .* opposite_main_diag;
sum(sum(opposite_diag_values));

% Other operations with matrices.
sum(a);  % sum of all values in a
prod(a);  % multiplication of all values in a
floor(a);  % round down
ceil(a);  % round up

% find() function.
find(a < 3);  % returns indexes of values less than 3
find(A < 3);  % returns indexes of ROWS less than 3
[r, c] = find(A >= 7);  % returns separated pairs (value from r, value from c).
