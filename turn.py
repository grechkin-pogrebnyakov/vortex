#!/usr/bin/python

import sys
from math import sin, cos, radians
import numpy as np

if len(sys.argv) < 4:
    print ("usage: {0} in_file out_file angle".format(sys.argv[0]))
    exit(1)
in_file = sys.argv[1]
out_file = sys.argv[2]
ang = float(sys.argv[3])
matr = np.array([[cos(radians(ang)), -sin(radians(ang))],[sin(radians(ang)), cos(radians(ang))]])
inf = open(in_file, "r")
outf = open(out_file, "w")
for line in inf:
    line = line.strip()
    rr = line.split()
    if len(rr) < 16:
        print >> outf, line
        continue
    try:
        n = int(rr[0])
        left = np.array([float(rr[1]), float(rr[2])])
        right = np.array([float(rr[3]), float(rr[4])])
        contr = np.array([float(rr[5]), float(rr[6])])
        birth = np.array([float(rr[7]), float(rr[8])])
        norm = np.array([float(rr[9]), float(rr[10])])
        tang = np.array([float(rr[11]), float(rr[12])])
        length = float(rr[13])
        n_left = int(rr[14])
        n_right = int(rr[15])
        new_left = matr.dot(left)
        new_right = matr.dot(right)
        new_contr = matr.dot(contr)
        new_birth = matr.dot(birth)
        new_norm = matr.dot(norm)
        new_tang = matr.dot(tang)
        new_line = "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}".format(
                n, new_left[0], new_left[1], new_right[0], new_right[1], new_contr[0], new_contr[1],
                new_birth[0], new_birth[1], new_norm[0], new_norm[1], new_tang[0], new_tang[1], length, n_left, n_right)
        print >> outf, new_line
     
    except ValueError:
        print >> outf, line
        continue


