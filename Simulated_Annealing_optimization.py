# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 22:06:02 2018

@author: erdbrca
"""

import numpy as np
import time
from random import choice

def costf(x):
    q = np.array(x)
    return np.sum((q - 1)**2)

def annealingoptimize(domain, costf, T=10000, cool=.97, step=1):
    # random initialisation
    vec = [float(np.random.randint(domain[i][0], domain[i][1])) for i in range(len(domain))]
    
    unit_directions = [-1,1]
    it = 0
    while T>.1:
        i = np.random.randint(0, len(domain)-1)  # choose an index
        dir = step*(-1)**choice(unit_directions)  # choose a direction to change it

        # Create a new list with one of the values changed
        vecb = vec[:]
        vecb[i] += dir
        if vecb[i]<domain[i][0]: 
            vecb[i] = domain[i][0]
        elif vecb[i]>domain[i][1]:
            vecb[i] = domain[i][1]
            
        # Calculate current cost and new cost
        ea = costf(vec)
        eb = costf(vecb)

        # Is it better or does it make the probability cut-off?
        if eb<ea:
            vec = vecb  # new solution is better
        else:
            prob = np.exp((-eb - ea) / T)
            if np.random.rand()<prob:
                vec = vecb  # new solution is NOT better but nevertheless selected
        
        # Decrease the temperature
        T *= cool
        it += 1
    return vec, it
    
        
if __name__=='__main__':
    domain = [(0,9)]*10
    SA_solution, i = annealingoptimize(domain, costf, step=.5)  # See cost function: solution is [0,1,2,..,9]
    print(SA_solution)
