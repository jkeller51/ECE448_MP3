# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 15:12:41 2018

@author: Haokun Li
"""

import os
import numpy as np
import pylab as plt

class Network(object):
    def __init__(self, hidden_layers=(256, 256, 256), in_dim=5, 
                 out_dim=3, bias=True, activation='relu', lr=1e-1, epoch=500, 
                 batch_size=100):
        """ Neural Network object, fully connected.
        
        Args:
            hidden_layers(tuple): how many neurons in each layer
            in_dim(int): input dimention
            out_dim(int): output dimention
            bias(boolean): whether or not to add bias
            activation(str): activation function
            lr(float): learning rate
            epoch(int): how many epochs
            batch_size(int): batch size
        Returns:
            (None)
        """
        self.hidden_layers = hidden_layers
        self.layers = len(hidden_layers)
        self.bias = bias
        self.activation = activation
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        
        # Initialize W
        self.W = []
        for count in range(self.layers + 1):
            if count == 0:
                if self.bias:
                    input_dim_ = in_dim + 1
                else:
                    input_dim_ = in_dim
                output_dim_ = self.hidden_layers[count]
            elif count == self.layers:
                if self.bias:
                    input_dim_ = self.hidden_layers[count-1] + 1
                else:
                    input_dim_ = self.hidden_layers[count-1]
                output_dim_ = 3
            else:
                if self.bias:
                    input_dim_ = self.hidden_layers[count-1] + 1
                else:
                    input_dim_ = self.hidden_layers[count-1]
                output_dim_ = self.hidden_layers[count]            
            w_ = np.random.uniform(low=0.0, high=0.01, size=(input_dim_, output_dim_))
            self.W.append(w_)
                
        self.dW = []
        self.dB = []
        
        self.Z = []
        self.dZ = []
        
        self.A = []
        self.dA = []
        
        self.X = None
        self.y = None
        
    def _shuffle_(self, X, y):
        """ Shuffle. """
        row, col = X.shape
        idx = list(range(row))
        np.random.shuffle(idx)
        X_ = X[idx]
        y_ = y[idx]
        return X_, y_
    
    def _relu_(self, X):
        """ Relu function.
        
        Args:
            X(np.array): matrix
        Returns:
            new_X(np.array): matrix
        """
        new_X = np.where(X<=0, 0.0, X)
        return new_X
    
    def _derivative_relu(self, X):
        """ Relu function.
        
        Args:
            X(np.array): matrix
        Returns:
            new_X(np.array): matrix
        """
        new_X = np.where(X<=0, 0.0, 1.0)
        return new_X
    
    def _forward(self, X):
        """ Forward propogation
        
        Args:
            X(np.array): matrix
        Returns:
            f(np.array): matrix
        """
        self.X = X.copy()
        self.Z = []
        self.A = [self.X]
        myinput = X.copy()
        for layer in range(0, self.layers+1):
            # Add bias column
            r, c = myinput.shape
            bias_column = np.ones((r, 1))
            if self.bias:
                myinput = np.append(myinput, bias_column, axis=1)
            
            # Matrix multiplication
            Z = np.matmul(myinput, self.W[layer])
            self.Z.append(Z)
            
            # Activation function
            if layer != self.layers:
                A = self._relu_(Z)
            else:
                A = Z.copy()
            self.A.append(A)

            # Next layer            
            myinput = A.copy()
        
        f = self.A[-1]
        return f
    
    def _loss(self, f, y):
        """ Compute loss.
        
        Args:
            f(np.array): matrix, returned by self._forward()
            y(np.array): vector, true labels
        Returns:
            loss(float)
            dF(np.array): matrix
        """
        n, dim = f.shape
        
        loss = 0.5/n * np.sum(np.square(f - y))
        dF = 1/n * (f - y)
        return loss, dF
    
    def _backward(self, f, y):
        """ Backward propogation.
        
        Args:
            f(np.array): matrix, returned by self._forward()
            y(np.array): vector, true labels
        Returns:
            (None)
        """
        self.dZ = []
        self.dA = []
        self.dW = []
        self.dB = []
        loss, dF = self._loss(f, y)
        
        temp = dF.copy()
        for count in range(self.layers, -1, -1):
            # Get dZ
            if count == self.layers:
                dZ = temp.copy()
            elif (self.bias == True):
                dZ = np.multiply(temp[:, 0:-1], self._derivative_relu(self.A[count+1]))
            elif (self.bias == False):
                dZ = np.multiply(temp, self._derivative_relu(self.A[count+1]))
            self.dZ.insert(0, dZ)
            
            # Get dW, dB, previous dA
            dA = np.matmul(dZ, self.W[count].transpose())
            self.dA.insert(0, dA)
            dW = np.matmul(self.A[count].transpose(), dZ)
            self.dW.insert(0, dW)
            dB = np.sum(dZ, axis=0)
            self.dB.insert(0, dB)
            
            # For previous layer
            temp = dA.copy()
    
    def _update(self):
        """ Update weights.
        
        Args:
            (None)
        Returns:
            (None)
        """
        for count in range(self.layers+1):
            if self.bias == True:
                self.W[count][0:-1, :] -= self.lr * self.dW[count]
                self.W[count][-1, :] -= self.lr * self.dB[count]
            else:
                self.W[count][:, :] -= self.lr * self.dW[count]
    
    def save_weights(self):
        """ Save weights to files.
        
        Args:
            (None)
        Returns:
            (None)
        """
        # Structure
        filename = "./q_weights/structure.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as myfile:
            myfile.write(' '.join(str(x) for x in self.hidden_layers))
        
        # Parameters
        filename = "./q_weights/parameters.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as myfile:
            myfile.write('bias,{0}\n'.format(self.bias))
            myfile.write('activation,{0}\n'.format(self.activation))
            myfile.write('epoch,{0}\n'.format(self.epoch))
        
        # Weights
        for i in range(len(self.W)):
            filename = "./q_weights/weight_{0}.txt".format(i)
            np.savetxt(filename, self.W[i])
        
        # Print info
        os.path.dirname('..')
        print('Weights successfully saved.')
    
    def load_weights(self):
        """ Load weight matrices from files.
        
        Args:
            (None)
        Returns:
            (None)
        """
        # Structure
        structure = np.genfromtxt('./q_weights/structure.txt', dtype=np.int16)
        self.hidden_layers = tuple(structure)
        self.layers = len(self.hidden_layers)
        
        # Parameters
        filename = "./q_weights/parameters.txt"
        with open(filename, 'r') as myfile:
            for line in myfile:
                temp = line.strip().split(',')
                if temp[0] == 'bias':
                    self.bias = bool(temp[1])
                elif temp[0] == 'activation':
                    self.activation = temp[1]
                elif temp[0] == 'epoch':
                    self.epoch = int(temp[1])
                else:
                    print('Unknown argument: {0}'.format(temp[0]))
        
        # Weights
        self.W = []
        for i in range(self.layers+1):
            myfile = './q_weights/weight_{0}.txt'.format(i)
            w_ = np.genfromtxt(myfile)
            self.W.append(w_)
        
        # Log info
        print('Successfully loaded!')
    
    def train(self, X, y):
        """ Training.
        
        Args:
            X(np.array): matrix
            y(np.array): vector
        Returns:
            (None)
        """
        loss_list = []
        
        # Shuffle at the beginning of each epoch
        for i in range(self.epoch):
            row, col = X.shape
            X, y = self._shuffle_(X, y)
            
            # Train batch
            for idx in range(row // self.batch_size):
                start = idx * self.batch_size
                end = (idx + 1) * self.batch_size                
                X_ = X[start: end, :]
                y_ = y[start: end]
                
                f = self._forward(X_)
                loss, dF = self._loss(f, y_)
                self._backward(f, y_)
                self._update()
            
            loss, _ = self._loss(f, y_)
            print('epoch:{0:3d}\tloss:{1:.6f}'.format(i, loss))
            loss_list.append(loss)
            
        # Plot training curve
        plt.plot(range(self.epoch), loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('deep_q_learning_curve.png', dpi=1600)
        print('Learning curve saved.')
    
    def predict(self, X):
        """ Training.
        
        Args:
            X(np.array): matrix or vector, test data
        Returns:
            predict(np.array or int): vector or scalar
        """
        f = X.copy()
        for ct in range(self.layers+1):
            # Add bias
            if len(f.shape) == 2:
                r, c = f.shape
                bias_column = np.ones((r, 1))
                if self.bias == True:
                    f = np.append(f, bias_column, axis=1)
            elif self.bias == True:
                f = np.append(f, 1)
            
            f = np.matmul(f, self.W[ct])
            if ct != self.layers:
                f = self._relu_(f)
        
        predict = f
        return predict