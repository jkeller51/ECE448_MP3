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

    i = 0
    while (i < result.shape[0]):
        if (i + 200) >= result.shape[0]:
            temp_list = result[i::]
        else:
            temp_list = result[i:i+200]
        
        mean_list.append(np.mean(temp_list))
        
        i += 200
    
    N = len(mean_list)
    plt.plot(range(N), mean_list, 'b-', label='mean')
    plt.legend()
    plt.show()