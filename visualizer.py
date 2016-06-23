#!/usr/bin/env python
import os
import matplotlib
disp = os.environ.get('DIAPLAY')
if disp is None or disp == '':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import animation
import os.path
import argparse

def is_folder(string):
    if not os.path.isdir(string):
        msg = "%r is not a directory" % string
        raise argparse.ArgumentTypeError(msg)
    return string

parser = argparse.ArgumentParser(description='Create vortex animation.')#, prog='visualizer')
parser.add_argument('-fi', '--first_input', type=is_folder, required=True, help='path to first component files', metavar='/path/to/first_input', dest='first_folder')
parser.add_argument('-si', '--second_input', type=is_folder, required=True, help='path to second component files', metavar='/path/to/second_input', dest='second_folder')
parser.add_argument('-o', '--output', default=None, help='if set animation will be saved to file', metavar='outfile.mp4', dest='output')
parser.add_argument('-s', '--step', default=1, type=int, help='step of input files', metavar='S', dest='step')
parser.add_argument('-p', '--profile', default=None, type=file, help='profile file', metavar='path/to/profile', dest='profile')
parser.add_argument('--start', default=0, type=int, help='index of first file', metavar='S', dest='start')
parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1', help='print program version')
parser.add_argument('--verbose-debug', help='print program version')
arguments = vars(parser.parse_args())

pr = arguments.get('profile')
pr_arr = []
if pr:
    for line in pr:
        rr = line.strip().split()
        if len(rr) < 15:
            continue
        pr_arr.append([float(rr[1]),float(rr[2])])        

folder = []
def parse_folder(param_name):
    folder.append(arguments.get(param_name))
    index = len(folder) - 1;
    if folder[index][len(folder[index])-1] != '/':
        folder[index] += '/'

parse_folder('first_folder')
parse_folder('second_folder')
output = arguments.get('output', None)

step = arguments.get('step')
start = arguments.get('start')

fnames = [[],[]]
i = start
while True:
    fname = "Kadr%06d.txt" % i
    first_path = folder[0] + fname
    second_path = folder[1] + fname
    if (not os.path.exists(first_path)) or (not os.path.exists(second_path)):
        break
    i += step
    fnames[0].append(first_path)
    fnames[1].append(second_path)

print "got %u frames" % len(fnames[0])

fig = plt.figure()
fig.set_figwidth(10.67)
#fig.set_figwidth(10)
#fig.set_size_inches(12, 6)
ax = plt.axes(xlim=(-1, 5), ylim=(-1, 1) )
point_size = 2 if output is None else 1.3
dash, = ax.plot([], [], 'r.', ms=point_size)
dash1, = ax.plot([], [], 'b.', ms=point_size)
profile = matplotlib.patches.Polygon(pr_arr, lw=1, fc='none')
ax.add_patch(profile)
ax.set_aspect('equal')

def init():
    dash.set_data([],[])
    dash1.set_data([],[])
    profile.set_edgecolor('none')
    return dash, dash1, profile

def animate(k):
    x = []
    y = []
    f = open( fnames[0][k] )
    for line in f:
        line = line.strip().split(' ')
        if len(line) == 1:
            continue
        x.append(float(line[2]))
        y.append(float(line[3]))
    f.close()
    dash.set_data(x, y)
    x1 = []
    y1 = []
    f = open( fnames[1][k] )
    for line in f:
        line = line.strip().split(' ')
        if len(line) == 1:
            continue
        x1.append(float(line[2]))
        y1.append(float(line[3]))
    f.close()
    dash1.set_data(x1, y1)
    profile.set_edgecolor('k')
    return dash, dash1, profile

anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=len(fnames[0]), interval=10)

if output is not None:
    anim.save(output, writer='ffmpeg', dpi=180)#, codec='libx264')
else:
    plt.show()
