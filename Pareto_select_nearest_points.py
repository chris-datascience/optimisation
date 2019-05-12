# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:10:41 2016

@author: Kris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
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

#def Mahalanobis_distances(xx, yy):  # to do: use this instead of Euclidean dist. see what happens
#   # Source:  https://stackoverflow.com/questions/27686240/calculate-mahalanobis-distance-using-numpy-only#27691752
#    X = np.vstack([xx,yy])
#    V = np.cov(X.T)
#    VI = np.linalg.inv(V)
#    delta = xx - yy
#    D = np.sqrt(np.einsum('nj,jk,nk->n', delta, VI, delta))
#    return D
#    return np.diag(np.sqrt(np.dot(np.dot((xx-yy),VI),(xx-yy).T)))

def min_distance_to_Pareto(data_pt, Pareto_pts):
    return np.min(np.linalg.norm(Pareto_pts - data_pt, axis=1))

def get_points_closest_to_Pareto(X, Y, maxX=True, maxY=False, n_select=20):
    XY = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
    X_, Y_ = list(X), list(Y)
    
    xy, Pareto_inds = pareto_frontier(X_, Y_, maxX, maxY)
    Pareto_pts = np.array(xy)
    
    distances_to_pareto = np.empty((len(X),))
    for i,pt in enumerate(XY):
        distances_to_pareto[i] = min_distance_to_Pareto(pt, Pareto_pts)

    # Make selection of n_select closest Pareto pts
    closest_inds = np.argsort(distances_to_pareto)[:n_select]
    closest_pts = np.sort(XY)[:n_select]
    return Pareto_pts, Pareto_inds, closest_pts, closest_inds


if __name__=='__main__':
    N = 500
    X = 30.+np.random.randn(N,1)*10
    Y = np.random.randn(N,1)*20

    DF = pd.DataFrame(data=np.hstack((X,Y)), columns=['X','Y'])
    DF['X_norm'] = (DF.X - DF.X.mean()) / (DF.X.max()- DF.X.min())
    DF['Y_norm'] = (DF.Y - DF.Y.mean()) / (DF.Y.max()- DF.Y.min())
    
    
    Pareto_pts, Pareto_inds, closest_pts, closest_inds = get_points_closest_to_Pareto(DF.X_norm.values, DF.Y_norm.values, True, False, 40)
    
    plt.figure(figsize=(10,7))
    plt.plot(DF.X_norm, DF.Y_norm, 'k.')
    plt.plot(Pareto_pts[:,0], Pareto_pts[:,1], 'bd-', alpha=.4)   
    plt.plot(closest_pts[:,0], closest_pts[:,1], 'mo', alpha=.3)
    

#    DF = DF.sort_values('dist_to_Pareto', ascending=True)
#    selected_pts = DF[['X_norm','Y_norm']][:n_select].values
#    plt.plot(selected_pts[:,0], selected_pts[:,1], 'co', alpha=.4)
    