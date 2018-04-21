# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:37:26 2018

@author: Haokun Li
"""

import numpy as np
import pylab as plt

if __name__ == '__main__':
    result = np.genfromtxt('sarsa_log.txt', delimiter='\n', dtype=np.int16)
    mean_list = []
    x = []

    i = 200
    while (i < result.shape[0]):
        temp_list = result[0:i]        
        mean_list.append(np.mean(temp_list))
        x.append(i)        
        i += 200
    
    N = len(mean_list)
    plt.plot(x, mean_list, 'b-', label='mean')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Score')
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('mean_score.png', dpi=1600)