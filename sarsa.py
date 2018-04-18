# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:23:55 2018

@author: Haokun Li
"""

from graphics import *
from pong_model import *
import numpy as np

class SARSA(object):
    def __init__(self, PongEnvironment, Window):
        # Environment
        self.environment = PongEnvironment
        self.window = Window
        
        # Allow to explore
        self.explore = 1e4
        self.threshold = 2e3
        self.step_count = 0
        
        # Discrete position
        self.ball_x = None
        self.ball_y = None
        self.ball_velocity_x = None
        self.ball_velocity_y = None
        self.paddle_y = None
        
        # Used to discrete position
        min_ = 0
        max_ = (400 - 80) / 400
        self.bins_paddle = np.arange(min_, max_+0.0001, (max_-min_)/11)
        self.bins_ball = np.arange(0, 1.0001, 1/11)
        
        # Some important variables
        self.alpha = 1
        self.gamma = 0.8
        
        # Implemented as a matrix
        # shape (ballx, bally, ballvelox, ballveloy, paddley, action)
        self.q_table = self.load_q_table('sarsa_q_table.csv')
        self.count_table = np.zeros(shape=(12, 12, 2, 3, 12, 3), dtype=np.int64)
        
        # Reward after some action
        self.reward = None
        
        # Actions to take
        self.actions = [1, 0, -1]
        self.action = None
        self.action_ = None
        
        # State, in the form of tuples
        self.current_state = self._get_current_state_()
        self.current_state_q = 0
        self.next_state = None
    
    def _discretize_positions_(self):
        """ Create 12x12 grid and mapping positions into these grids
        
        Args:
            (None)
        Returns:
            (None)
        """
        mylist = [self.environment.ball_x, 
                  self.environment.ball_y]
        self.ball_x, self.ball_y = np.digitize(mylist, self.bins_ball)
        self.paddle_y = np.digitize([self.environment.paddle_y], self.bins_paddle)[0]
        
        # np.digitize returns indices counting from 1 (instead of 0)
        # so we need to extract 1 here
        self.ball_x -= 1
        self.ball_y -= 1
        self.paddle_y -= 1
        
        # update positions in the environment
        self.environment.ball_x = self.bins_ball[self.ball_x]
        self.environment.ball_y = self.bins_ball[self.ball_y]
        self.environment.paddle_y = self.bins_paddle[self.paddle_y]
    
    def _discretize_velocity_(self):
        """ Making velocities discrete
        
        Args:
            (None)
        Returns:
            (None)
        """
        if self.environment.ball_velocity_x > 0:
            self.ball_velocity_x = 1
            self.environment.ball_velocity_x = self.bins_ball[1]
        else:
            self.ball_velocity_x = -1
            self.environment.ball_velocity_x = self.bins_ball[1] * -1
        
        if self.environment.ball_velocity_y > 0.015:
            self.ball_velocity_y = 1
            self.environment.ball_velocity_y = self.bins_ball[1]
        elif self.environment.ball_velocity_y < -0.015:
            self.ball_velocity_y = -1
            self.environment.ball_velocity_y = self.bins_ball[1] * -1
        else:
            self.ball_velocity_y = 0
            self.environment.ball_velocity_y = 0
    
    def _get_current_state_(self):
        """ Get current state."""
        self._discretize_positions_()
        self._discretize_velocity_()
        return (self.ball_x, self.ball_y, self.ball_velocity_x, 
                self.ball_velocity_y, self.paddle_y)
    
    def _choose_action_(self):
        """ Get next action, allow to explore
        
        Args:
            (None)
        Returns:
            action_idx(int)
        """
        state = self._get_current_state_()
        if self.step_count < self.explore:
            return np.random.randint(low=-1, high=2)
        else:
            count_list = self.count_table[state]
            not_threshold_idx = np.argwhere(count_list<=self.threshold).reshape(-1,)
            if len(not_threshold_idx) < 3:
                return np.random.choice(not_threshold_idx)
            else:
                action_temp = self.q_table[state]
                return np.argmax(action_temp)
                
    def _take_action_(self, idx):
        """ Take selected action, and observe reward
        
        Args:
            idx(int)
        Returns:
            (None)
        """
        # Dummy, forced discretize
        _ = self._get_current_state_()
        
        # Take action
        a = self.actions[idx]
        self.paddle_y = self.paddle_y + a
        if self.paddle_y >= 11:
            self.paddle_y = 11
        elif self.paddle_y <= 0:
            self.paddle_y = 0
        
        # Update environment
        pos = self.bins_paddle[self.paddle_y]
        self.environment.paddle_y = pos
        _ = self._get_current_state_()
        
        # Get reward
        original_score = self.environment.score
        self.environment.update(self.window)
        new_score = self.environment.score
        if self.environment.lost == True:
            self.reward = -1
        else:
            self.reward = new_score - original_score
    
    def _update_q_table_(self):
        """ Update Q table.
        
        Args:
            (None)
        Returns:
            (None)
        """
        q_s_a = self.current_state + (self.action,)
        q_sprime_aprime = self.next_state + (self.action_,)
        
        # Update count table
        self.count_table[q_s_a] += 1
        
        current_state_q = self.q_table[q_s_a]
        self.current_state_q = current_state_q
        if self.reward == -1:
            next_state_q = 0
        else:
            next_state_q = self.q_table[q_sprime_aprime]
        
        temp = self.reward + self.gamma * next_state_q - current_state_q
        self.q_table[q_s_a] = current_state_q + self.alpha * temp
    
    def save_q_table(self, path):
        """ Save Q table
        
        Args:
            path(str)
        Returns:
            (None)
        """
        self.q_table.tofile(path, sep=',', format='%.6f')
    
    def load_q_table(self, path):
        """ Load Q table from file
        
        Args:
            path(str)
        Returns:
            q_table(np.array): in the shape of (12, 12, 2, 3, 12, 3)
        """
        try:
            q_table = np.genfromtxt(path, delimiter=',')
            q_table = q_table.reshape((12, 12, 2, 3, 12, 3))
            print('Q table loaded!')
        except IOError:
            print('Q table not found!')
            q_table = np.zeros(shape=(12, 12, 2, 3, 12, 3), dtype=np.float32)
        return q_table
    
    def train(self):
        """ Training.
        Args:
            (None)
        Returns:
            (None)
        """
        while True:
            # Avoid raising errors when window is closed
            if self.window._open == False:
                break
            
            myscore = 0
            self.current_state = self._get_current_state_()
            self.action = self._choose_action_()            
            
            while self.environment.lost == False:
                # Avoid raising errors when window is closed
                if self.window._open == False:
                    break
                
                # Decay alpha
                self.alpha = 8e5 / (1e6 + self.step_count)
                if self.alpha <= 0.5:
                    self.alpha = 0.5
                
                # Save Q table
                if (self.step_count % 10000 == 0):
                    self.save_q_table('sarsa_q_table.csv')
                    print('Q table successfully saved!')
            
                # Main loop
                print('Step:{0:6d}\tQ-value:{1:.3f}\tScore:{2}\tAlpha:{3:.4f}' \
                    .format(self.step_count, self.current_state_q, myscore, self.alpha))
                self._take_action_(self.action)            
                self.next_state = self._get_current_state_()
                self.action_ = self._choose_action_()
                self._update_q_table_()
                self.current_state = self.next_state
                self.action = self.action_
            
                # If lost, reset the game and start over
                # else, count how many scores our agent has erned
                if self.reward == -1:
                    self.environment.reset()
                    break
                else:
                    myscore += self.reward
            
                # Counting steps
                self.step_count += 1
                
            # Keep recording scores in every epoch
            with open('log.txt', 'a') as f:
                f.write('{0}\n'.format(myscore))