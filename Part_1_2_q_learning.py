# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:02:49 2018

@author: Shiratori
"""

import graphics as gfx
import pong_model as pm
from q_learning import QLearning

if __name__ == '__main__':
    # Set up environment
    environment = pm.PongModel(0.5, 0.5, -0.03, 0.01, 0.5, paddleX=0) #initial state changed; defending line on the left. also, paddle now on left, so paddleX changed
    window = gfx.GFX(wall_x=1, player_x=0) #switched layout
    window.fps = 9e16
    
    # Set up model
    model = QLearning(environment, window, C=5e3, gamma=0.99, explore = -1,
                      threshold=-1, log=True, log_file='q_test_log_(1_1)_A.txt', 
                      mode='test', q_table_file='q_q_table_(1_1)_A.csv')
    
    # Training
    model.train()
