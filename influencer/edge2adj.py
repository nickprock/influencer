"""
Created on Thu Mar  04 18:40:07 2021

@author: nico
"""

import numpy as np

def edge2adj(edgelist: list, dim: int, weight: bool = True, direct: bool = True):
    """
    Basic edgelist to adjiacency matrix function

    Parameters
	-------------------
    edgelist: a list

    dim: int. The matrix shape

    weight: bool. default True. If True is a weighted graph, else not weighted.

    direct: bool. default True. If True is a direct graph, else undirect connections.

    Returns
    -------------------
    adj: a numpy array

    Example
    -------------------
    >> import numpy as np
    >> from influencer.edge2adj import edge2adj
    >> edgelist = [[0,1], [1,2], [0,3], [2,1], [3,1]]
    >> adj = edge2adj(edgelist = edgelist, dim = 4)
    """
    adj = np.zeros((dim, dim), dtype=int)
    for i in range(len(edgelist)):
        if direct:
            if weight:
                adj[edgelist[i][0], edgelist[i][1]] += 1
            else:
                adj[edgelist[i][0], edgelist[i][1]] = 1
        else:
            if weight:
                adj[edgelist[i][0], edgelist[i][1]] += 1
                adj[edgelist[i][1], edgelist[i][0]] += 1
            else:
                adj[edgelist[i][0], edgelist[i][1]] = 1
                adj[edgelist[i][1], edgelist[i][0]] = 1
    return adj