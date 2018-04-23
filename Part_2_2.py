# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 23:12:38 2018

@author: Haokun Li
"""

import pong_model as pm
import graphics as gfx
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

def game(N, window, environment, nn):
    """ Training.
    Args:
        N(int): N test games
        window(gfx.GFX)
        environment(pong_model.PongModel)
        nn(neural_network.Network)
    Returns:
        score(float): average score
    """
    episode_ct = 0
    total_score = 0
    
    while (episode_ct < N):
        # Avoid raising errors when window is closed
        if window._open == False:
            break
        
        # Initialize a new game
        environment.reset()
        myscore = 0
        
        while not environment.lost:
            # Get current state the best action
            current_state = np.array([environment.ball_x, environment.ball_y,
                                      environment.ball_velocity_x, 
                                      environment.ball_velocity_y,
                                      environment.paddle_y])
            action = np.argmax(nn.predict(current_state))
            if action == 0:
                environment.move_up()
            elif action == 2:
                environment.move_down()
            
            environment.update(window)
            if environment.score == -1:
                break
            else:
                myscore = environment.score
            
        # Counting epicodes
        episode_ct += 1
        total_score += myscore
        print(total_score)
        
    # Close the window
    window.close()
    return total_score / N

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
    print()
    
    # Play game
    environment = pm.PongModel(0.5, 0.5, 0.03, 0.01, 0.4)
    window = gfx.GFX()
    window.fps = 3e16
    avg_score = game(1000, window, environment, nn)