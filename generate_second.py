import sys
n_of_points = 101
n_of_lines = 5
x = -0.56
if len(sys.argv) > 1:
    x = float(sys.argv[1])
y_min = -2.0
y_max = 2.0
dy = (y_max - y_min)/(n_of_points-1)
ddy = dy/n_of_lines
print n_of_points*n_of_lines
for j in range(0, n_of_lines):
    for i in range(0, n_of_points):
        y = y_min + j * ddy + i * dy
        print('{0} 0.008 {1} {2} 0.0 0.0 0.0 100'.format(i + j * n_of_points, x + j * ddy, y));
