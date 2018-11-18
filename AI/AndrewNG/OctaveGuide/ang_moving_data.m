A = [
  1 2;
  3 4;
  5 6
];

sz = size(A);
sz_rows = size(A, 1);  % Size of first dimension
sz_cols = size(A, 2);  % Size of the second dimension
% disp(sz);
% disp(sz_rows);
% disp(sz_cols);

v = [1 2 3 4];
longest_dimension_v = length(v);  % 4
longest_dimension_A = length(A);  % 3

% load data file: x1 x2 y
load data1.dat  % load('data1.dat')
% load data2.dat  % load('data2.dat')

% clear data1  % removes 'data1' variable from octave session

% who  % shows variables inside current octave session
% whos  % gives detail view about variables (with sizes, bytes, class)

size(data1);

first_ten_entries = data1(1:10)

% save myfile.mat first_ten_entries  % save data to disc

% my_pi = pi
% save myfile.txt my_pi -ascii  % save as text (ASCII)
