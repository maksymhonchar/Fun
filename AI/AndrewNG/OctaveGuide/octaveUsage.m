% Execute N commands one after another:
a=1; b=2; c=3;

% Disable last warning in Octave.
[text, id] = lastwarn();
warning('off', id)

% Format how output (disp() or a simple output) will be displayed.
format long
format short

% Documentation for the function: [help function_name].
help eye
help help

% Set uo Octave search path.
addpath('C:\Users\username\...');

% Octave session variables.
save myfile.mat first_ten_entries  % save data to disc
load data1.dat  % same as load('data1.dat')
clear data1  % removes 'data1' variable from octave session
who  % shows variables inside current octave session
whos  % gives detail view about variables (with sizes, bytes, class)

% Save data on the disc.
save myfile.txt variable_name -ascii  % save as text (ASCII)
print -dpng 'angPlot.png'  % save figure as png

% Exit current session.
exit
