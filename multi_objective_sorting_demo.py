# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:58:16 2018

@author: Kris
"""

"""
    << Fast non-dominated sort >>
    Non-dominated sorting algorithm for sorting/ranking of a given population of samples according to multiple objectives.    
    See original NSGA-ii paper <<A fast and elitist multiobjective genetic algorithm: NSGA-II>> by Deb et al. 2002

    Goal is to identify all non-dominated Pareto fronts (front 0 is Pareto optimal front).
    Notation follows definitions in above paper.
"""

import numpy as np
import matplotlib.pyplot as plt


#Np = 0  # domination count; 0 means solution is not dominated by any other solution (i.e. on Pareto front)
#Nq is number of solutions dominating q
#p = 0  # number of solutions domination this solution
#Sp = set()  # set of solutions that this solution dominates
#Q = set()  # second nondominated front set

# P is solution population; p is solution under scrutiny; q is any other solution


# --> Going to minimise everyting, i.e. every dimension!
# SO NEED TO BE ABLE TO REPHRASE EACH PROBLEM INTO THIS FORMAT!


def is_dominant(x, y):
    # If lower in all dimensions implies dominating
    return all(x - y < 0) 


# == GENERATE TEST DATA ==
n_samples = 20
n_dims = 2
P = np.random.randn(n_samples, n_dims)

# === MAIN ===
N = {}  # Domination count: number of solutions which dominate p
rank = {}
F = {}  # Front sets
F[1] = set()
Sp = {}
for i,p in enumerate(P):
    Sp[i] = set()
    N[i] = 0
    for qi in [j for j in range(P.shape[0]) if j!=i]:
        if is_dominant(p, P[qi]):   # True if p dominates q
            Sp[i].add(qi)  # Add q to set of solutions dominated by p
            #print('%s dominates %s' % (str(list(p)), str(list(P[qi]))))
        elif is_dominant(P[qi], p):  # NOTA BENE: CANNOT USE "else" HERE!!!
            #print(qi)
            N[i] += 1  # increment domination counter of p (+1 everytime p is not dominant)
    #print(i, N[i])
    if N[i]==0:  # if p is not dominated by anyone
        #print('%i added to 1st front.' % i)
        rank[i] = 1  # ..then p belongs to first front
        F[1].add(i)

# Now expand to further non-dominated fronts:
ii = 1
while F[ii]:
    Q = set()
    for i in F[ii]:
        for q in Sp[i]:
            N[q] -= 1
            if N[q]==0:
                rank[q] = ii + 1
                Q.add(q)
    ii += 1
    F[ii] = Q
F = {k:list([P[x] for x in v]) for k,v in F.items() if len(v)>0} 

# === Plotting ===
plt.figure(figsize=(8,8))
colors = 'gmybrc'*99
for front in F.keys():
    leg = True
    for f in F[front]:
        if leg:
            plt.plot(f[0], f[1], colors[front-1]+'o', markersize=12, alpha=.6, label=str(front))
            leg = False
        else:
            plt.plot(f[0], f[1], colors[front-1]+'o', markersize=12, alpha=.6)
# Equivalent, but harder to get legend right:
#for f, front in rank.items():
#    p0, p1 = P[f]
#    plt.plot(p0, p1, colors[front-1]+'o', markersize=12, alpha=.6)    

         
plt.plot(P[:,0], P[:,1], 'k.', alpha=.9)
plt.legend(fontsize=14)