# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:32:27 2018

@author: jkell
"""
import numpy as np

sgn = lambda inp: np.sign(inp)
lin = lambda inp: inp

class HiddenLayer:
    """ Layer of perceptrons
    """
    def __init__(self, size, activate=sgn, bias=True):
        self.nextLayer = None
        self.size=size
        self.activate = activate
        self.bias = bias
        
        
class NeuralNetwork:
    def __init__(self, inSize, outSize):
        self.hlayers = []
        self.outputLayerSize = outSize
        self.inputLayerSize = inSize
        self.weights = None
        self.inputBias = True
        self.outputactivation = lin;
        
    def add_hidden_layer(self, size, activation=None, bias=True):
        if (activation != None):
            thislayer = HiddenLayer(size, activation, bias)
        else:
            thislayer = HiddenLayer(size, bias=bias)
        if (len(self.hlayers) > 0):
            self.hlayers[len(self.hlayers)-1].nextLayer = thislayer
        self.hlayers.append(thislayer)
        
        
    def generate_weights(self, mag=1):
        """ Generate init weights for this neural network
        weights will be randomized
        
        Args:
            mag : multiply the initialized weights by this (to avoid overflows)
        """
        
        if (len(self.hlayers) == 0):
            print("No weights to generate! Please add hidden layers.")
            return
        
        weights = []
        
        insize = self.inputLayerSize
        if (self.inputBias == True):
            insize += 1
            
        layersize = self.hlayers[0].size
        if (self.hlayers[0].bias == True):
            layersize += 1
            
        weights.append(np.random.randn(insize, layersize)*mag)
        
        for i in range(0, len(self.hlayers)-1):
            layer1size = self.hlayers[i].size
            if (self.hlayers[i].bias == True):
                layer1size+=1
                
            layer2size = self.hlayers[i+1].size
            if (self.hlayers[i+1].bias == True):
                layer2size+=1
                
            weights.append(np.random.randn(layer1size, layer2size))
        
        layer1size = self.hlayers[len(self.hlayers)-1].size
        if (self.hlayers[len(self.hlayers)-1].bias == True):
            layer1size+=1
        weights.append(np.random.randn(layer1size, self.outputLayerSize))
        
        self.weights = weights

    def forward(self, X):
        """ forward propogation of data X
        """
        X = np.asarray(X)
        
        if (len(X.shape) == 1):
            X = np.reshape(X, (1, len(X)))
            
        if (self.inputBias == True):
            # add a bias unit to each row
            rows = []
            
            for i in range(0, X.shape[0]):
                rows.append(np.append(X[i],1))
                
            X = np.asarray(rows)
        
        
        if (len(self.hlayers) == 0):
            print("No hidden layers yet! Please add hidden layers.")
            return 0
        
        z = np.matmul(X, self.weights[0]) # result of inputlayer x weights
        a = self.hlayers[0].activate(z) # apply activation function at first hidden layer
        
        if (len(self.hlayers) > 1):
            for i in range(1, len(self.hlayers)):
                z = np.matmul(a, self.weights[i])
                a = self.hlayers[i].activate(z)
                
        # output layer
        z = np.matmul(a, self.weights[len(self.weights)-1])
        a = self.outputactivation(z)
        
        return a
                
        
        