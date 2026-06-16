"""
Created on Thu Mar  04 18:40:07 2021

@author: nico
"""

import numpy as np


def edge2adj(edgelist: list, dim: int, weight: bool = True, direct: bool = True) -> np.ndarray:
    """
    Basic edgelist to adjacency matrix function.

    Parameters
    -------------------
    edgelist: a list of [src, dst] pairs

    dim: int. The matrix shape (dim x dim).

    weight: bool. default True. If True, repeated edges increase the weight.

    direct: bool. default True. If True the graph is directed; otherwise undirected.
        For undirected graphs, self-loops are counted once (not twice).

    Returns
    -------------------
    adj: numpy array of shape (dim, dim)

    Example
    -------------------
    >> from influencer.edge2adj import edge2adj
    >> edgelist = [[0,1], [1,2], [0,3], [2,1], [3,1]]
    >> adj = edge2adj(edgelist=edgelist, dim=4)
    """
    adj = np.zeros((dim, dim), dtype=int)
    for edge in edgelist:
        src, dst = edge[0], edge[1]
        if direct:
            adj[src, dst] = adj[src, dst] + 1 if weight else 1
        else:
            adj[src, dst] = adj[src, dst] + 1 if weight else 1
            if src != dst:  # self-loops are a single edge: count once, not twice
                adj[dst, src] = adj[dst, src] + 1 if weight else 1
    return adj
