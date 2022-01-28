"""
Created on Fri Jan  28 09:56:07 2022

@author: nico
"""

import torch

def hits(adjMatrix, p: int = 100):
	
  n = adjMatrix.shape[0]
  a = torch.ones([1,n])
  h=torch.ones([1,n])
  pa = a
  authority = {}
  hub = {}

  for k in range(1,p):
    h1 = torch.mm(adjMatrix, pa.T)/torch.norm(torch.mm(adjMatrix, pa.T))
    a1 = torch.mm(adjMatrix.T, h1)/torch.norm(torch.mm(adjMatrix.T , h1))

    h = torch.vstack((h,torch.mm(adjMatrix, a[k-1,:].T)/torch.norm(torch.mm(adjMatrix, a[k-1,:].T))))
    a = torch.vstack((a,torch.mm(adjMatrix.T, h[k,:].T)/torch.norm(torch.mm(adjMatrix.T, h[k,:].T))))

    pa = a1.T

    for i in range(n):
      authority[str(i)] = a[-1,i]
      hub[str(i)] = h[-1,i]
  return hub, authority, h, a