%% Part 1: matrices, vectors.

% The ; denotes we are going back to a new row.
A = [
	1,2,3;
	4,5,6;
	7,8,9;
	10,11,12
]

% Initialize a vector.
v = [
	1;
	2;
	3;
]

% Get the dimension of the matrix A where m=rows n=columns.
[m, n] = size(A)
dim_A = size(A)

% Get the dimension of the vector v.
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A.
A_23 = A(2,3)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Part 2: addition, subtraction of matrices
A = [
	1,2,3;
	4,5,6
]

B = [
	7,8,9;
	10,11,12
]

s_const = 2

% Element-wise addition/subtraction.
add_AB = A + B
sub_AB = A - B

% Scalar multiplication/division.
mult_As = A * s_const
div_As = A / s_const

% Try to add a matrix and scalar: addition to each matrix element s_const.
add_As = A + s_const

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Part 3: matrix_x_vector/matrix_x_matrix multiplication

A = [
  1,2,3;
  4,5,6;
  7,8,9
]

v =  [
  1;
  2;
  3;
]

mult_Av = A * v

B = [
  10;
  11;
  12
]

mult_AB = A * B

a = [
  1;
  2;
  3;
]

I = eye(3)

mult_IA = I * a

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Part 4: inverse and transpose

A = [
  1,2,0;
  0,5,6;
  7,0,9
]

A_trans = A' 
% inv, pinv
A_inv = inv(A)
 
A_invA = inv(A) * A

%%%%%%

u = [
  1;
  3;
  -1
]
v = [
  2;
  2;
  4
]
u_trans = u'
mult_utrans_v = u_trans * v

% try to inverse things, again:
A = [
  1,2,3;
  4,5,6;
  7,8,9
]

inv_A = pinv(A)
