"""
Created on Wed Feb  5 15:02:07 2020

@author: nico
"""

def hits(adjMatrix, p: int = 100, np: bool = True):
    
    if (np):
        import numpy as np
    else:
        import jax.numpy as np
    
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


def tophits(T, epsilon: float = 0.001, np: bool = False):
    
    if (np):
        import numpy as np
    else:
        import jax.numpy as np
    
    u, v, w = np.empty([1,T.shape[0]]), np.empty([1,T.shape[1]]), np.empty([1,T.shape[2]])
    sigma = []

    x = np.ones((T.shape[0], 1))
    y = np.ones((T.shape[1], 1))
    z = np.ones((T.shape[2], 1))
    
    tx = np.squeeze(x)
    ty = np.squeeze(y)
    tz = np.squeeze(z)
    
    lambda0=100
    continua=True
    num=1
    
    while(continua):
        x1 = np.tensordot(T,ty,axes=([1],[0]))
        x = np.squeeze(np.tensordot(x1,tz,axes=([1],[0])))
        
        y1 = np.tensordot(T,x,axes=([0],[0]))
        y = np.squeeze(np.tensordot(y1,tz,axes=([1], [0])))
        
        z1 = np.tensordot(T,x,axes=([0],[0]))
        z = np.squeeze(np.tensordot(z1,y,axes=([0],[0])))
        
        tx=x/np.linalg.norm(x)
        ty=y/np.linalg.norm(y)
        tz=z/np.linalg.norm(z)
        
        lambda1 = np.linalg.norm(tx)*np.linalg.norm(ty)*np.linalg.norm(tz)
        
        if(abs(lambda1-lambda0) < epsilon):
            continua=False
        
        lambda0 = lambda1
        num = num+1
    
    u = np.vstack((u,tx))
    v = np.vstack((v,ty))
    w = np.vstack((w,tz))
    
    sigma.append(lambda1)
    
    u = u[1:,:]
    v = v[1:,:]
    w = w[1:,:]

    return u, v, w
