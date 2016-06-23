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
import numpy as np

def is_folder(string):
    if not os.path.isdir(string):
        msg = "%r is not a directory" % string
        raise argparse.ArgumentTypeError(msg)
    return string

parser = argparse.ArgumentParser(description='Create vortex animation.')#, prog='visualizer')
parser.add_argument('-in', '--input', type=is_folder, required=True, help='path to input directory', metavar='/path/to/kadrs', dest='input_folder')
parser.add_argument('-o', '--output', default=None, help='if set animation will be saved to file', metavar='outfile.mp4', dest='output')
parser.add_argument('-s', '--step', default=1, type=int, help='step of input files', metavar='S', dest='step')
parser.add_argument('-p', '--profile', default=None, type=file, help='profile file', metavar='path/to/profile', dest='profile')
parser.add_argument('--start', default=0, type=int, help='index of first file', metavar='S', dest='start')
parser.add_argument('--limit', default=0, type=float, help='minimum drawable point', metavar='L', dest='limit')
parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1', help='print program version')
parser.add_argument('--verbose-debug', help='use verbose debug')
arguments = vars(parser.parse_args())

pr = arguments.get('profile')
pr_arr = []
if pr:
    for line in pr:
        rr = line.strip().split()
        if len(rr) < 15:
            continue
        pr_arr.append([float(rr[1]),float(rr[2])])        

input_folder = arguments.get('input_folder')
if input_folder[-1] != '/':
    input_folder += '/'

output = arguments.get('output', None)

step = arguments.get('step')
start = arguments.get('start')
lim = arguments.get('limit')

fnames = []
i = start
while True:
    fname = "Kadr%06d.txt" % i
    path = input_folder + fname
    if not os.path.exists(path):
        break
    i += step
    fnames.append(path)

print "got %u frames" % len(fnames)

fig = plt.figure()
fig.set_figwidth(10.67)
#fig.set_figwidth(10)
#fig.set_size_inches(12, 6)
ax = plt.axes(xlim=(-1, 5), ylim=(-1, 1) )
point_size = 2 if output is None else 1.0
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
    x1 = []
    y1 = []
    f = open( fnames[k] )
    for line in f:
        line = line.strip().split(' ')
        if len(line) == 1:
            continue
        g = float(line[7])
        if g > lim:
            x.append(float(line[2]))
            y.append(float(line[3]))
        elif g < -lim:
            x1.append(float(line[2]))
            y1.append(float(line[3]))
    f.close()
    dash.set_data(x, y)
    dash1.set_data(x1, y1)
    profile.set_edgecolor('k')
    return dash, dash1, profile

anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=len(fnames), interval=100)

if output is not None:
    anim.save(output, writer='ffmpeg', dpi=180, codec='libx264')
else:
    plt.show()
