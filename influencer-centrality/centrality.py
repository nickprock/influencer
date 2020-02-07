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


def tophits(T, epsilon: float = 0.001):
    
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

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:14:54 2020

@author: integris
"""

import numpy as np

def socialAU(mu, mi, mw, T, epsilon: float = 0.001):
    u, v, w = np.empty([1,T.shape[0]]), np.empty([1,T.shape[1]]), np.empty([1,T.shape[2]])
    
    x=np.ones((T.shape[0],1))
    y=np.ones((T.shape[1],1))
    z=np.ones((T.shape[2],1))

    tx = np.squeeze(x)
    ty = np.squeeze(y)
    tz = np.squeeze(z)

    lambda0=10
    continua=True
    
    ##########################
    # user
    #nU=mu.shape[0]
    aU=np.ones([1,mu.shape[0]])
    hU=np.ones([1,mu.shape[0]])
    paU=aU
    phU=hU
    #########################
    # item
    #nI=mi.shape[0]
    aI=np.ones([1,mi.shape[0]])
    hI=np.ones([1,mi.shape[0]])
    paI=aI
    phI=hI
    #########################
    # word
    # nW=mw.shape[0]
    aW=np.ones([1,mw.shape[0]])
    hW=np.ones([1,mw.shape[0]])
    paW=aW
    phW=hW
    #########################
    num=1
    k=1
    
    while (continua):
        
        # user
        h1U = np.dot(mu, paU.T)/np.linalg.norm(np.dot(mu, paU.T))
        a1U = np.dot(mu.T, h1U)/np.linalg.norm(np.dot(mu.T , h1U))
        hU = np.vstack((hU,np.dot(mu, aU[k-1,:].T)/np.linalg.norm(np.dot(mu, aU[k-1,:].T))))
        aU = np.vstack((aU,np.dot(mu.T, hU[k,:].T)/np.linalg.norm(np.dot(mu.T, hU[k,:].T))))
        paU = a1U.T
        phU = h1U.T
        
        # item
        h1I = np.dot(mi, paI.T)/np.linalg.norm(np.dot(mi, paI.T))
        a1I = np.dot(mi.T, h1I)/np.linalg.norm(np.dot(mi.T , h1I))
        hI = np.vstack((hI,np.dot(mi, aI[k-1,:].T)/np.linalg.norm(np.dot(mi, aI[k-1,:].T))))
        aI = np.vstack((aI,np.dot(mi.T, hI[k,:].T)/np.linalg.norm(np.dot(mi.T, hI[k,:].T))))
        paI = a1I.T
        phI = h1I.T
        
        # word
        h1W = np.dot(mw, paW.T)/np.linalg.norm(np.dot(mw, paW.T))
        a1W = np.dot(mw.T, h1W)/np.linalg.norm(np.dot(mw.T , h1W))
        hW = np.vstack((hW,np.dot(mw, aW[k-1,:].T)/np.linalg.norm(np.dot(mw, aW[k-1,:].T))))
        aW = np.vstack((aW,np.dot(mw.T, hW[k,:].T)/np.linalg.norm(np.dot(mw.T, hW[k,:].T))))
        paW = a1W.T
        phW = h1W.T
        
        x1 = np.tensordot(T,ty,axes=([1],[0]))
        x = np.squeeze(np.tensordot(x1,tz,axes=([1],[0])))
    
        y1 = np.tensordot(T,x,axes=([0],[0]))
        y = np.squeeze(np.tensordot(y1,tz,axes=([1], [0])))
    
        z1 = np.tensordot(T,x,axes=([0],[0]))
        z = np.squeeze(np.tensordot(z1,y,axes=([0],[0])))
        
        # user
        h2U=hU[k,:]
        a2U=aU[k,:]
        SNP=(h2U+a2U)
        SNP=np.squeeze(SNP.T)
    
        # item
        a2I=aI[k,:]
        a2I=np.squeeze(a2I.T)
    
        # word
        a2W=aW[k,:]
        a2W=np.squeeze(a2W.T)
        
        #################################
        txx=(x/np.linalg.norm(x))+SNP
        tx=txx/np.linalg.norm(txx)
    
        tyy=(y/np.linalg.norm(y))+a2I
        ty=tyy/np.linalg.norm(tyy)
    
        tzz=(z/np.linalg.norm(z))+a2W
        tz=tzz/np.linalg.norm(tzz)
        ################################
        
        lambda1 = np.linalg.norm(tx)*np.linalg.norm(ty)*np.linalg.norm(tz)
        
        if(abs(lambda1-lambda0) < epsilon):
            continua=False
            
        lambda0 = lambda1
        num+=1
        k+=1
    
    u = np.vstack((u,tx))
    v = np.vstack((v,ty))
    w = np.vstack((w,tz))
    
    sigma = lambda1
    
    u = u[1:,:]
    v = v[1:,:]
    w = w[1:,:]
    
    return u, v, w, sigma
