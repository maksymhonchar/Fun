data = [ 0 : 0.01: 0.98 ];  % [0.00:0.98] with step 0.01
size(data);  % 1 99
y1 = sin(2 * pi * 4 * data);  % also  an array
y2 = cos(2 * pi * 4 * data);
plot(y1);
hold on;  % to place both functions on the same plot.
plot(y2, 'r');
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('ang plot')

print -dpng 'angPlot.png'  % save figure as a file. Not only png could be file format

% close;  % close a figure

% Plot on different windows:
figure(2); plot(data, y1);
figure(3); plot(data, y2);

% Plot in the same window, but on two rectangles.
subplot(1, 2, 1);  % Divides plot a 1x2 grid, access first element
plot(data, y1);  % fill up 1x2 grid, on the left side
subplot(1, 2, 2);  % Access SECOND element in 1x2 grid
plot(data, y2);  % fill up 1x2 grid, on the right side

% Change axis scales of LAST plot (subplot, right side)
axis( [ 0.5 1 -5 5 ] )  % x:from 0.5 to 1; y:from -1 to 1

clf;  % clear last figure

% Visualize a matrix:
A = magic(5);
imagesc(A);  % grid of colors

imagesc(A), colorbar, colormap gray;  % also put colorbar to show what color corresponds to

imagesc(magic(100)), colorbar, colormap gray; 


% execute N commands one after another:
a=1; b=2; c=3;
