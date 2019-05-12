# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:58:16 2018

@author: Kris
"""

"""
    << Fast non-dominated sort for pandas dataframe>>
    Non-dominated sorting algorithm for sorting/ranking of a given population of samples according to multiple objectives.    
    See original NSGA-ii paper <<A fast and elitist multiobjective genetic algorithm: NSGA-II>> by Deb et al. 2002

    Goal is to identify all non-dominated Pareto fronts (front 0 is Pareto optimal front).
    Notation follows definitions in above paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def is_dominant(x, y, minim, maxim):
    """ Dominance is defined per dimension by minmax, a list of booleans.
        'True' in minmax means minimisation, False means maximisation.
        In other words, this dominance test consists of testing whether each entry 
        of x the dimension of which we want to minimise is smaller than the corresponding entry in y
        and vice versa for dimensions we wish to maximise.
    """
    return all(x[minim] - y[minim] < 0) * all(x[maxim] - y[maxim] > 0)
    #return all(x - y < 0) 


def perform_sorting(df0, minimise=None, plotting=False):
    """
        Dataframe should have ordered row indexes 0...len(df)-1.
        'minimise' is of booleans indicating for each column what the objective is: to minimise or maximise;
        default is to minimise everything (True).
        Outputs:
            - 'rank' is dict of ranked indexes giving frontier number for each sample row; lower front meaning 'better'.
            - 'F' is dict of ranked sample points giving all samples for each identified frontier.
    """
#    # Approach 1: Flip columns that need to be maximised
#    if len(maximise)!=len(df.columns) or not bool(maximise):
#        print('\nminimise argument has wrong format.')
#        return
#    df = df.copy()
#    for col in df.columns[maximise]:
#        df[col] = df[col].max() - df[col].values
    
    original_index = list(df0.index)
    df = df0.copy()
    if minimise is None:
        minimise = [True] * len(df.columns)
    maximise = [False if f else True for f in minimise]
    
    # -- Initialise --
    df.index = list(range(len(df)))
    P = df.values  # solutions population to minimise along each dimension
    N = {}  # Domination count: number of solutions which dominate p
    rank = {}
    F = {}  # Front sets
    F[1] = set()
    Sp = {}
    # === MAIN ===
    for i,p in enumerate(P):
        Sp[i] = set()
        N[i] = 0
        for qi in [j for j in range(P.shape[0]) if j!=i]:
            if is_dominant(p, P[qi], minimise, maximise):   # True if p dominates q
                Sp[i].add(qi)  # Add q to set of solutions dominated by p
                #print('%s dominates %s' % (str(list(p)), str(list(P[qi]))))
            elif is_dominant(P[qi], p, minimise, maximise):  # NOTA BENE: CANNOT USE "else" HERE!
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

    if len(df.columns)==2 and plotting:
        # === Plotting ===
        plt.figure(figsize=(8,8))
        colors = 'gmybrck'*99
        for front in F.keys():
            leg = True
            for f in F[front]:
                if leg:
                    plt.plot(f[0], f[1], colors[front-1]+'o', markersize=12, alpha=.5, label=str(front))
                    leg = False
                else:
                    plt.plot(f[0], f[1], colors[front-1]+'o', markersize=12, alpha=.5)
        plt.plot(P[:,0], P[:,1], 'k.', alpha=.9)
        # TO DO: draw lines between points in colour of frontier
        plt.legend(fontsize=13, title='frontier', loc=1, fancybox=True, framealpha=.5)
        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.title('Pareto optimal frontiers', fontsize=15)
        
    # Add rankings to dataframe, mapping back to original index:
    index_rank = {original_index[k]:v for k,v in rank.items()}
    Pareto = pd.DataFrame.from_dict(data=index_rank, orient='index')
    df_ranked = df0.copy().join(Pareto, how='left').rename(columns={0:'Pareto'})
    return F, rank, df_ranked
 

# TO DO: SEE HOW SORTING A RATE (e.g. Profit per volume) RELATES TO PARETO-SORTING OF SEPARATE QUANTITIES (e.g. minimise volume, maximise profit)


if __name__=='__main__':
    # == GENERATE TEST DATA ==
    n_samples = 25
    n_dims = 2
    DF = pd.DataFrame(data=np.random.randn(n_samples, n_dims), columns=list('ABCDEFGHIJKLMN')[:n_dims])
    
    ranked_samples, ranked_indexes, ranked_df = perform_sorting(DF, minimise=[False, True], plotting=True)
    
    
    # --- POST PROCESSING EXAMPLE (see Merchant_Ranking_Scores.py) ---
    """ [1] Ranking on Pareto dominance
        Sources:
            NSGA-II Multiobjective genetic algorithm paper by Deb et al.(2002)
            https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf
        NB. Not using NSGA-II, only the 'fast non-dominated sort' algorithm described in the paper.
        
        Pareto optimisation is aimed at maximising Sales volumes and GP (GrossProfit), 
        plus a posteriori sorting on Chargeback rates.
    
    Suppose df_Overview has this format:
         ID   Good_tx  Bad_tx  Profit     Bad_rate   Profit_rate
    0    24     22          0   28486.5      0.0    1238.543478
    1    33      3          0     462.0      0.0     115.500000
    2    43      1          0      48.6      0.0      24.300000

    Then can do:
    _, ranked_indexes = perform_sorting(df_Overview[['Bad_rate', 'Profit_rate']], \
                                        minimise=[True, False], plotting=False)
    Pareto = pd.DataFrame.from_dict(data=ranked_indexes, orient='index')
    df_ranking = df_Overview.join(Pareto, how='left') \
                              .rename(columns={0:'Pareto'}) \
                              .sort_values(by=['Pareto','Bad_rate'], ascending=[True, True])
    show_ranking(df_ranking)
    """