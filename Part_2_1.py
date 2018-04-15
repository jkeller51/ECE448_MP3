# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:37:40 2018

@author: jkell
"""

import DNN
import graphics as gfx
import time
import numpy as np

def _ReLu(z):
    outp = np.zeros(z.shape)
    if len(z.shape) == 2:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if (z[i,j] > 0):
                    outp[i,j] = z[i,j]
    else:
        for i in range(z.shape[0]):
            if (z[i] > 0):
                outp[i] = z[i]
    return outp

ReLu = lambda inp: _ReLu(inp)   # rectified linear

# 0 = move paddle up
# 1 = don't move
# 2 = move paddle down

if __name__ == '__main__':
    
    window = gfx.GFX()
    
    Network = DNN.NeuralNetwork(5,3)   # 5 inputs, 3 outputs
    Network.add_hidden_layer(10, activation=ReLu)
    Network.generate_weights()
    
    X = [0.5,0.5,0.1,0.2,0.5]
    
    print(Network.forward(X))
    
    while 1:
        # main loop
        if (window._open == False):
            break
        window.update() # this happens on a different thread
        
        
        
    
        

    print("Game ended.")

    

