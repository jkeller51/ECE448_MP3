# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:36:57 2018

@author: jkell
"""

import numpy as np

resolution = 10
import random

def intersect_line(x1,y1,x2,y2,linex):
    """
    Determine where the line drawn through (x1,y1) and (x2,y2)
    will intersect the line x=linex
    """    
    if (x2 == x1):
        return 0
    return ((y2-y1)/(x2-x1))*abs(linex-x1)+y1

#ball_x = np.asarray(range(0,resolution))/(resolution)
#ball_y = ball_x.copy()
#
#ball_vx = np.asarray(range(0,resolution))/resolution * 0.06 - 0.03
#ball_vy = ball_vx.copy()
#
#paddle_y = np.asarray(range(0,resolution))/(resolution)
#
#data = []
#
#for bx in ball_x:
#    for by in ball_y:
#        for vx in ball_vx:
#            for vy in ball_vy:
#                for py in paddle_y:
##                if (vx < 0):  # ball moving away
#                    if (by < py-0.01):
#                        yy = 0
#                    
#                    elif (by > py+0.01):
#                        yy = 2
#                        
#                    else:
#                        yy = 1
##                    else:
##                        y_int = intersect_line(bx,by,vx,vy,1)
##                        if (y_int < py+0.1):
##                            yy = 0
##                        elif (y_int == py+0.1):
##                            yy = 1
##                        elif (y_int > py+0.1):
##                            yy = 2
#                    data.append([bx, by, vx, vy, py, yy])
    
data = []
for _ in range(100000):
    bx = random.random()
    by = random.random()
    vx = random.random() * 0.06 - 0.03
    vy = random.random() * 0.06 - 0.03
    py = random.random()
    
    if (by+vy < py):
        yy = 0
    
    elif (by+vy > py+0.2):
        yy = 2
        
    else:
        yy = 1
        
    data.append([bx, by, vx, vy, py, yy])
                    
f = open('./data/test_policy_easy.txt', 'w')
for d in data:
    f.write(str(d[0]) + " " + str(d[1]) + " " + str(d[2]) + " " + str(d[3]) + " " + str(d[4]) + " " + str(d[5]) + "\n")
    
f.close()