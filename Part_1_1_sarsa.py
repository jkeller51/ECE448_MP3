# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:21:28 2018

@author: Shiratori
"""

import graphics as gfx
import pong_model as pm
from sarsa import SARSA

if __name__ == '__main__':
    # Set up environment
    environment = pm.PongModel(0.5, 0.5, 0.03, 0.01, 0.4, paddleX=1)
    window = gfx.GFX(wall_x=0, player_x=1)
    window.fps = 9e16
    
    # Set up model
    model = SARSA(environment, window, alpha=0.5, gamma=0.99, explore=-1, threshold=-1, 
                  log=True, log_file='test_sarsa_log.txt', mode='test')
    
    # Training
    model.train()
