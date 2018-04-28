# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:36:57 2018

@author: jkell
"""

import numpy as np
ball_x = np.asarray(range(0,8))*0.1
ball_y = ball_x.copy()

ball_vx = np.asarray(range(0,8))*0.06 - 0.03
ball_vy = ball_vx.copy()

paddle_y = np.asarray(range(0,8))*0.1

data = []

for bx in ball_x:
    for by in ball_y:
        for vx in ball_vx:
            for vy in ball_vy:
                for py in paddle_y:
                    if (by < py):
                        yy = 0
                    elif (by == py):
                        yy = 1
                    elif (by > py):
                        yy = 2
                    data.append([bx, by, vx, vy, py, yy])
                    
f = open('./data/test_policy.txt', 'w')
for d in data:
    f.write(str(d[0]) + " " + str(d[1]) + " " + str(d[2]) + " " + str(d[3]) + " " + str(d[4]) + " " + str(d[5]) + "\n")
    
f.close()