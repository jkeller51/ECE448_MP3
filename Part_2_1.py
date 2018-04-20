# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:37:40 2018

@author: jkell
"""

import DNN
import graphics as gfx
import time
import numpy as np
import random
import pong_model as png

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

def _derivative_ReLU(z):
    outp = np.zeros(z.shape)
    if len(z.shape) == 2:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if (z[i,j] > 0):
                    outp[i,j] = 1
    else:
        for i in range(z.shape[0]):
            if (z[i] > 0):
                outp[i] = 1
    return outp


ReLu = lambda inp: _ReLu(inp)   # rectified linear

# 0 = move paddle up
# 1 = don't move
# 2 = move paddle down

def load_data(fname):
    f = open(fname, 'r')
    data = []
    y=[]
    for line in f:
        linedata = line.split(" ")
        
        state = [float(linedata[0]), float(linedata[1]), float(linedata[2]),
                 float(linedata[3]), float(linedata[4])]
        y.append(float(linedata[5]))
        data.append(state)
        
    return data, y
        

if __name__ == '__main__':
    
    BATCH_SIZE = 1
    
    Network = DNN.NeuralNetwork(5,3)   # 5 inputs, 3 outputs
    Network.add_hidden_layer(16, activation=ReLu)
    Network.add_hidden_layer(16, activation=ReLu)
    Network.generate_weights(1)
    
    X = [0.5,0.5,0.03,0.01,0.4]
    
    
    # load data
    training_data, y = load_data("./data/expert_policy.txt")
    
    

    print("Training Neural Network...")
    
    for _ in range(100):
        
        # generate a new batch
        batch = []
        batch_y = []
        for _ in range(BATCH_SIZE):
            selection = random.randint(0,len(training_data)-1)
            while (training_data[selection] in batch):
                selection = random.randint(0,len(training_data)-1)
            batch.append(training_data[selection])
            batch_y.append(y[selection])
            
        
        # train
        a = Network.forward(batch)
        Network.backward(batch_y[0])
        print("dW", Network.dW)
        loss = Network.update_weights(0.1)
        print("L:", loss)
        print(Network.forward([training_data[0], training_data[1000], training_data[2000]]))
    
    print("Training Done. Loss =",loss)
    a = Network.forward(training_data)
    
    window = gfx.GFX()
    Game = png.PongModel(0.5, 0.5, 0.03, 0.01, 0.4)   # initialize state
    
    Lflag = False   # lost flag
    while 1:
        # main loop
        if (window._open == False):
            break
        Game.update(window)   # this happens on a different thread
        
        state = Game.get_state()
        actionlist = Network.forward(state)
        maxidx = 0
        for i in range(1,len(actionlist)):
            if (actionlist[i] > actionlist[maxidx]):
                maxidx = i
        if (maxidx == 0):
            Game.move_up()
        elif (maxidx == 2):
            Game.move_down()
        
        if (Game.lost == True and Lflag == False):
            Game.reset()  
        

    print("Game ended.")

    

