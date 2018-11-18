% for-loop:
v = zeros(10, 1);
for i=1:10,
  v(i) = 2^i;
  v(i) *= 10;
endfor;

Indices = 1:10;
for i=Indices,
  disp(i);
endfor;

% while, break, continue keywords example
i = 1;
while true,
  v(i) = 999;
  i = i+1;
  if i == 6,
    break
  endif;
endwhile;

v(1) = 2;
if v(1) == 1,
  disp('The value is one');
elseif v(1) == 2,
  disp('The value is two');
else
  disp('The value is neither one or two')
endif;

exit
