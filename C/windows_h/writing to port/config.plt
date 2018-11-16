set title "Values from COM3 port"
set datafile separator ","
set xlabel "Time, sec"
set ylabel "Values"
plot "plot.dat" using 1:2 title 'values' with lines
pause 1
replot
reread