# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:32:27 2018

@author: jkell
"""
import numpy as np

sgn = lambda inp: np.sign(inp)
lin = lambda inp: inp

def vectorize(z):
    if (len(z.shape) == 1):
        z = np.reshape(z, (1, len(z)))
    return z

def cross_entropy(a, y):
    c=[0,0,0]
    for val in y:
        if val == 2:
            c[2]+=1
        elif val == 1:
            c[1]+=1
        elif val == 0:
            c[0]+=1
    a = np.squeeze(a)
    expa = np.exp(a)
    if (len(expa.shape) == 2):
        summ = np.sum(expa, axis=1)
    else:
        summ = np.sum(expa)
    L = 0
    da = np.zeros(a.shape)
    if (len(a.shape) == 2):
        # batch processing
        for i in range(len(y)):
            onef = np.zeros(np.size(a,1))
            L += (a[i,int(y[i])] - np.log(summ[i]))
            onef[int(y[i])] = 1
            da[i] = (onef-expa[i]/summ[i])
        L *= -1/(len(a))
        da *= -1/(len(a))
        
    else:
        # single input
        onef = np.zeros(a.shape)
        L = -1*(a[int(y)] - np.log(summ))
        onef[int(y)] = 1
        da = (-1)*(onef-expa/summ)
    # calculate gradient respect to output
    
    
    return L, da

#def _derivative_ReLU(z):
#    outp = np.zeros(z.shape)
#    if len(z.shape) == 2:
#        for i in range(z.shape[0]):
#            for j in range(z.shape[1]):
#                if (z[i,j] > 0):
#                    outp[i,j] = 1
#    else:
#        for i in range(z.shape[0]):
#            if (z[i] > 0):
#                outp[i] = 1
#    return outp
    
def _derivative_ReLU(da, z):
    outp = np.zeros(da.shape)
    if len(da.shape) == 2:
        for i in range(da.shape[0]):
            for j in range(da.shape[1]):
                if (da[i,j] != 0):
                    outp[i,j] = da[i,j]
    else:
        for i in range(da.shape[0]):
            if (da[i] != 0):
                outp[i] = da[i]
    return outp

def affine(X,weights,bias=True):
    if (bias == True): # the last weight of each column is the bias
        return np.matmul(X, weights[:-1,:]) + weights[-1,:]
    else:
        return np.matmul(X, weights)

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
        
        self.batchsize=0
        self.loss=0
        
        self.dW = []
        
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
        
        
        # input -> first layer
        insize = self.inputLayerSize
        if (self.inputBias == True):
            insize += 1
            
        layersize = self.hlayers[0].size
#        if (self.hlayers[0].bias == True):
#            layersize += 1
        
        
        weights.append(np.random.uniform(size=(insize, layersize))*mag)
        
        # DELETE? initialize biases to 0.
        weights[-1][-1,:] = 0
        
        for i in range(0, len(self.hlayers)-1):
            # layer i -> layer i+1
            layer1size = self.hlayers[i].size
            if (self.hlayers[i].bias == True):
                layer1size+=1
                
            layer2size = self.hlayers[i+1].size
#            if (self.hlayers[i+1].bias == True):
#                layer2size+=1
                
            weights.append(np.random.uniform(size=(layer1size, layer2size))*mag)
            # DELETE? initialize biases to 0.
            weights[-1][-1,:] = 0
        
        # last layer -> output
        layer1size = self.hlayers[len(self.hlayers)-1].size
        if (self.hlayers[len(self.hlayers)-1].bias == True):
            layer1size+=1
        weights.append(np.random.uniform(size=(layer1size, self.outputLayerSize))*mag)
        # DELETE? initialize biases to 0.
        weights[-1][-1,:] = 0
        
        self.weights = weights

    def forward(self, X):
        """ forward propogation of data X
        """
        X = np.asarray(X)
        
        if (len(X.shape) == 1):
            X = np.reshape(X, (1, len(X)))
            
#        if (self.inputBias == True):
#            # add a bias unit to each row
#            rows = []
#            
#            for i in range(0, X.shape[0]):
#                rows.append(np.append(X[i],1))
#                
#            X = np.asarray(rows)
        
        
        if (len(self.hlayers) == 0):
            print("No hidden layers yet! Please add hidden layers.")
            return 0
        
        a = []
        z = []
        
        a.append(X)
        
        z.append(affine(X, self.weights[0]))
        a.append(np.squeeze(self.hlayers[0].activate(z[-1]))) # apply activation function at first hidden layer
        
        if (len(self.hlayers) > 1):
            for i in range(1, len(self.hlayers)):
                z.append(affine(a[len(a)-1], self.weights[i]))
                a.append(np.squeeze(self.hlayers[i].activate(z[-1])))
                    
        # output layer
        z.append(affine(a[len(a)-1], self.weights[len(self.weights)-1]))
        a.append(np.squeeze(self.outputactivation(z[-1])))
        
        self.a = a  # cache for backprop
        self.z = z
        
        return a[len(a)-1]
                
    
    def backward(self, y):
        # y is the expected output
        loss, dF = cross_entropy(self.a[len(self.a)-1], y)
        self.loss += loss
        
        dW=[]
        
        # assume output activation is linear (no activation)
        dz=dF
        
        for i in range(0, len(self.hlayers)):
            idx = len(self.hlayers)-1-i
            if (self.hlayers[idx].bias == True):
                W = self.weights[len(self.weights)-1-i][0:-1,:]
            else:
                W = self.weights[len(self.weights)-1-i]
            da = np.matmul(vectorize(dz),np.transpose(W))
            if (len(self.a[idx+1].shape) == 1):
                self.a[idx+1] = np.reshape(self.a[idx+1], (1, len(self.a[idx+1])))
            dW.insert(0, np.matmul(np.transpose(self.a[idx+1]), vectorize(dz))/len(y))
            if (self.hlayers[idx].bias == True):
                dW[0] = np.vstack([dW[0], np.sum(vectorize(dz),axis=0)])   # add bias db
            dz = _derivative_ReLU(da, self.a[idx])
            
        # input to first hidden layer
        if (self.inputBias == True):
            W = self.weights[0][0:-1,:]
        else:
            W = self.weights[0]
        if (len(self.a[0].shape) == 1):
                self.a[0] = np.reshape(self.a[0], (1, len(self.a[0])))
        da = np.matmul(vectorize(dz), np.transpose(W))
        dW.insert(0, np.matmul(np.transpose(self.a[0]), vectorize(dz))/len(y))
        if (self.inputBias == True):
                dW[0] = np.vstack([dW[0], np.sum(vectorize(dz),axis=0)])
        
        self.batchsize+=1
        if (len(self.dW) > 0):
            self.dW += dW
        else:
            self.dW = dW
        
    def update_weights(self, alpha=0.01):
        self.loss/=self.batchsize
        loss= self.loss
        
        for i in range(len(self.weights)):
            self.weights[i] -= alpha*self.dW[i]
        
        # reset for next batch
        self.loss=0
        self.batchsize=0
        self.dW = []
        return loss
        