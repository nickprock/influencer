"""
Created on Wed Feb  5 15:02:07 2020

@author: nico
"""

import numpy as np

def hits(adjMatrix, p: int = 100):
    n = adjMatrix.shape[0]
    
    a = np.ones([1,n])
    h = np.ones([1,n])
    
    pa=a
    #ph=h
    
    authority = {}
    hub = {}
    
    for k in range(1,p):
        h1 = np.dot(adjMatrix, pa.T)/np.linalg.norm(np.dot(adjMatrix, pa.T))
        a1 = np.dot(adjMatrix.T, h1)/np.linalg.norm(np.dot(adjMatrix.T , h1))
    
        h = np.vstack((h,np.dot(adjMatrix, a[k-1,:].T)/np.linalg.norm(np.dot(adjMatrix, a[k-1,:].T))))
        a = np.vstack((a,np.dot(adjMatrix.T, h[k,:].T)/np.linalg.norm(np.dot(adjMatrix.T, h[k,:].T))))
    
        pa = a1.T
        #ph = h1.T
        
    for i in range(n):
        authority[str(i)] = a[-1,i]
        hub[str(i)] = h[-1,i]
    
    return hub, authority, h, a
