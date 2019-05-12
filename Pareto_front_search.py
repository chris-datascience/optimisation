# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:23:59 2018

@author: erdbrca
"""

"""
[Adapted version of http://oco-carbon.com/metrics/find-pareto-frontiers-in-python/]

[original text:]
Method to take two equally-sized lists and return just the elements which lie 
on the Pareto frontier, sorted into order.
Default behaviour is to find the maximum for both X and Y, but the option is
available to specify maxX = False or maxY = False to find the minimum for either
or both of the parameters.
"""

import numpy as np

def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
    myList = sorted([(Xs[i], Ys[i]) for i in range(len(Xs))], reverse=maxX)
    if maxX:
        myIndex = np.argsort(Xs)[::-1]
    else:
        myIndex = np.argsort(Xs)
# Removing Nones and corresponding indices        
#    newList = myList[:]
#    None_inds = [i for i,v in enumerate(myList) if v is None]
#    newList = [v for i,v in enumerate(myList) if i not in None_inds]    
#    newInds = [v for i,v in enumerate(myIndex) if i not in None_inds]    
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    p_front_ind = [myIndex[0]]
# Loop through the sorted list
    for ind,pair in zip(myIndex[1:], myList[1:]):
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
                p_front_ind.append(ind)
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
                p_front_ind.append(ind)
# Turn resulting pairs back into a list of Xs and Ys
#    p_frontX = [pair[0] for pair in p_front]
#    p_frontY = [pair[1] for pair in p_front]
#    return p_frontX, p_frontY, p_front_ind
    return p_front, p_front_ind


# -----------------------------------------------------------
# Alernative method using Numpy (possibly faster), see:
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
# --> NOT SURE IF/HOW THIS WORKS YET.

def is_pareto_efficient_dumb(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs>=c, axis=1))
    return is_efficient

# Yet another option:
def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient
                