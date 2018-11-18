A = [
  1 2;
  3 4;
  5 6
];
A_32 = A(3,2);  % Third row, second column of A matrix

%  ":" means every element along that row/column.
second_row = A(2,:);  % Second row of A.
second_column = A(:,2);  % Second column of A.

A([1 3], :);  % Get everything from first and third row; Get all data from column
A([1 3], 1);  % Get everything from 1st and 3rd row; Get data from 1 column


% Sophisticated assignment:
A;
A(:, 2) = [
  10;
  11;
  12
];
A;

size(A);
A = [A, [100; 101; 102]];  % Append another column vector to right
size(A);

A(:);  % Put all elements of A matrix into a single vector
size(A);  % 3 3
size(A(:));  % 9 1

A = [1 2; 3 4; 5 6];
B = [11 12; 13 14; 15 16];
A;
B;
C = [A B];  % Concatenating of the matrices.
C = [A, B];  % same as C=[A B]

C = [A; B];  % Put the B at the bottom
size(C);  % 6 2
