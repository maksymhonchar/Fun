% Create a histogram of data.
hist(w);
hist(w, 50);

% Visualize a matrix.
A = magic(5);
imagesc(A);  % grid of colors
imagesc(A), colorbar, colormap gray;  % Put colorbar for measuring data.
imagesc(magic(100)), colorbar, colormap gray;

% Plot functions.
plot(y1);  % Create a plot.
hold on;  % Place both functions on the same plot.
plot(y2, 'r');  % Place the second plot.

% Setup plot parameters: labels names, title, legend.
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('ang plot')

% Close a figure
close;
% Clear the last figure.
clf;

% Plot on different windows:
figure(2); plot(data, y1);
figure(3); plot(data, y2);

% Plot in the same window, but on two rectangles.
subplot(1, 2, 1);  % Divides plot a 1x2 grid, access first element
plot(data, y1);  % fill up 1x2 grid, on the left side
subplot(1, 2, 2);  % Access SECOND element in 1x2 grid
plot(data, y2);  % fill up 1x2 grid, on the right side

% Change axis scales of LAST mentioned plot.
axis( [ 0.5 1 -5 5 ] )  % x:from 0.5 to 1; y:from -1 to 1
