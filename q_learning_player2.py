# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 23:59:10 2018

@author: Haokun Li
"""

from graphics import *
from pong_model import *
import numpy as np

class QLearning(object):
    def __init__(self, PongEnvironment, Window, C=5e3, gamma=0.99, 
                 explore=3e4, threshold=1e2, log=True, log_file='q_log.txt', 
                 mode='train', q_table_file='q_q_table.csv'):
        """ SARSA model object. 
        Q table is automatically saved every 10,000 steps.
        
        Args:
            PongEnvironment(pong_model.PongModel)
            Window(graphics.GFX)
            C(float): learning rate decay factor, C / (C + N(s, a))
            gamma(float): learning rate
            explore(float): number of steps, during which an action is randomly
                            chosen
            threshold(float): N(s, a) threshold
            log(boolean): whether or not to log game results
            log_file(str): log file name (of game results)
            mode(str): when 'train', endless training;
                       else, recording 200 games.
            q_table_file(str): path for saving and loading Q table
        Returns:
            (None)
        """
        # Environment
        self.environment = PongEnvironment
        self.window = Window
        
        # Log info
        self.log = log
        self.log_file = log_file
        self.q_table_file = q_table_file
        self.mode = mode
        
        # Allow to explore
        self.explore = explore
        self.threshold = threshold
        self.epsilon = 0.15
        self.step_count = 0
        
        # Discrete position
        self.ball_x = None
        self.ball_y = None
        self.ball_velocity_x = None
        self.ball_velocity_y = None
        self.paddle_y = None
        
        # Used to discretize position
        min_ = 0
        max_ = (400 - 80) / 400
        self.bins_paddle = np.arange(min_, max_+0.0001, (max_-min_)/11)
        self.bins_ball = np.arange(0, 1.0001, 1/11)
        
        # Learning parameters
        self.C = C
        self.gamma = gamma
        
        # Implemented as a matrix
        # shape (ballx, bally, ballvelox, ballveloy, paddley, action)
        self.q_table = self.load_q_table(self.q_table_file)
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
        self.paddle_y = np.digitize([self.environment.paddle2_y], self.bins_paddle)[0]
        
        # np.digitize returns indices counting from 1 (instead of 0)
        # so we need to extract 1 here
        self.ball_x -= 1
        self.ball_y -= 1
        self.paddle_y -= 1
    
    def _discretize_velocity_(self):
        """ Making velocities discrete
        
        Args:
            (None)
        Returns:
            (None)
        """
        if self.environment.ball_velocity_x > 0:
            self.ball_velocity_x = 1
        else:
            self.ball_velocity_x = -1
        
        if self.environment.ball_velocity_y > 0.015:
            self.ball_velocity_y = 1
        elif self.environment.ball_velocity_y < -0.015:
            self.ball_velocity_y = -1
        else:
            self.ball_velocity_y = 0
    
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
            prob = np.random.uniform(low=0.0, high=1.0)
            if (len(not_threshold_idx) > 0) and (prob < self.epsilon):
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
        # Take action
        a = self.actions[idx]
        self.environment.paddle2_y += a * 0.04
        if self.environment.paddle2_y <= 0:
            self.environment.paddle2_y = 0
        elif self.environment.paddle2_y >= 0.8:
            self.environment.paddle2_y = 0.8
        
        # Get reward
#        original_score = self.environment.score
#        self.environment.update(self.window)
#        new_score = self.environment.score
#        if self.environment.lost == True:
#            self.reward = -1
#        else:
#            self.reward = new_score - original_score
    
    def _update_q_table_(self):
        """ Update Q table.
        
        Args:
            (None)
        Returns:
            (None)
        """
        q_s_a = self.current_state + (self.action,)
        q_sprime = self.next_state
        
        # Update count table
        self.count_table[q_s_a] += 1
        alpha = self.C / (self.C + self.count_table[q_s_a])
        if alpha <= 0.8:
            alpha = 0.8
        
        # Update Q table
        current_state_q = self.q_table[q_s_a]
        self.current_state_q = current_state_q
        if self.reward == -1:
            next_state_q = 0
        else:
            next_state_q = np.max(self.q_table[q_sprime])
        
        temp = self.reward + self.gamma * next_state_q - current_state_q
        self.q_table[q_s_a] = current_state_q + alpha * temp
    
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
        episode_ct = 0
        
        # If 'mode == train', then endless
        # else record 200 consecutive games
        while (self.mode == 'train') or (episode_ct < 200):
            # Avoid raising errors when window is closed
            if self.window._open == False:
                break
            
            myscore = 0
            self.current_state = self._get_current_state_()
            
            while self.environment.lost == False:
                # Avoid raising errors when window is closed
                if self.window._open == False:
                    break
                
                # Save Q table
                if (self.step_count % 10000 == 0):
                    self.save_q_table(self.q_table_file)
                    print('Q table successfully saved!')
            
                # Main loop
                self.action = self._choose_action_()
                self._take_action_(self.action)
                self.next_state = self._get_current_state_()
                if self.mode == 'train':
                    self._update_q_table_()
                self.current_state = self.next_state
            
                # If lost, reset the game and start over
                # else, count how many scores our agent has earned
                if self.reward == -1:
                    self.environment.reset()
                    break
                else:
                    myscore += self.reward
            
                # Counting steps
                self.step_count += 1
                
            # Keep recording scores in every epoch
            if self.log == True:
                with open(self.log_file, 'a') as f:
                    f.write('{0}\n'.format(myscore))
            
            # Counting episodes
            episode_ct += 1
            if episode_ct % 100 == 0:
                print('Episode:{0:10d}\tScore:{1}'.format(episode_ct, myscore))
        
        # Close the window
        try:
            self.window.close()
        except:
            pass