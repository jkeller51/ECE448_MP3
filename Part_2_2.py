# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 23:12:38 2018

@author: Haokun Li
"""

from neural_network import Network
import numpy as np

def load_data(path):
    """ Load data from path
    
    Args:
        path(str)
    Returns:
        X(np.array)
        y(np.array)
    """
    data = np.genfromtxt(path, delimiter=' ')
    X = data[:, 0:5]
    y = data[:, 5::]
    return X, y
    
def accuracy(nn, X, y):
    """ Compute accuracy and display it
    
    Args:
        nn(neural_network.Network)
        X(np.array)
        y(np.array)
    Returns:
        (None)
    """
    p = nn.predict(X)
    a = np.argmax(p, axis=1)
    action = np.argmax(y, axis=1)
    acc = np.sum(a==action) / len(y)
    print('Accuracy:{0:.4f}'.format(acc))

if __name__ == '__main__':
    # Load data
    X, y = load_data('./data/expert_q.txt')
    
    # Initialize the model
    nn = Network(lr=9e-2, epoch=1000, bias=True)
    try:
        nn.load_weights()
    except:
        print('WARNING: No pre-trained networks!')        
        # Training
        nn.train(X, y)
        nn.save_weights()
    
    # Display accuracy
    accuracy(nn, X, y)