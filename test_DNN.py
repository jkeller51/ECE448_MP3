# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:00:22 2018

@author: jkell
"""
import DNN
import random
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


BATCH_SIZE = 100

Network = DNN.NeuralNetwork(5,3)   # 2 inputs, 3 outputs
Network.add_hidden_layer(256, activation=ReLu, bias=True)
Network.add_hidden_layer(256, activation=ReLu, bias=True)
Network.add_hidden_layer(256, activation=ReLu, bias=True)
Network.add_hidden_layer(256, activation=ReLu, bias=True)
Network.generate_weights(0.01)
X=[]
Y=[]
# construct training set
# load data
X, Y = load_data("./data/expert_policy.txt")

print("Training Neural Network...")

for i in range(200):
    
    # generate a new batch
    batch = []
    batch_y = []
    for _ in range(BATCH_SIZE):
        selection = random.randint(0,len(X)-1)
        while (X[selection] in batch):
            selection = random.randint(0,len(X)-1)
        batch.append(X[selection])
        batch_y.append(Y[selection])
        
    
    # train
    a = Network.forward(batch)
    Network.backward(batch_y)
    dW = Network.dW
    W=Network.weights
    aa = Network.a
#    print("dW", Network.dW)
    loss = Network.update_weights(0.1)
    if (i%20 == 0):
        YY = np.argmax(a,axis=1)
        error = BATCH_SIZE-sum(np.equal(batch_y,YY))
        print(i, "epochs. Error:", error)

print("Training Done. Loss =",loss)
a = Network.forward(X)