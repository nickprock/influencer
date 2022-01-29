"""
Created on Fri Jan  28 09:56:07 2022

@author: nico
"""

import torch

def hits(adjMatrix, p: int = 100, device = 0):
  """
  Calculate the hub and authority score in a net

  Parameters
  -------------------
  adjMatrix: a torch tensor NxN.

  p: int. Default 100. The max iteration.

  device: Default 0. Set the GPU. If GPU is not available it is set to "cpu" automatically.

  Returns
  -------------------
  hub: dict. The hub score for each node in the net.
  authority: dict. The authority score for each node in the net.
  h: torch tensor Nxp. (Optional). The hub score for each node in the net for each algorithm's step.
  a: torch tensor Nxp. (Optional). The authority score for each node in the net for each algorithm's step.

  Example
  -------------------
  >> import torch
  >> import numpy as np
  >> from influencer.torch_centrality import hits

  Create an adjiacency matrix with numpy

  >> adjM = np.random.rand(10, 10)
  >> adjM[adjM>0.5]=1
  >> adjM[adjM<=0.5]=0
  >> MT = torch.from_numpy(adjM).float().to(0) 
  >> # .to(0) if GPU is available, test your GPU with torch.cuda.is_available()
  >> hub, aut, _, _ = hits(adjMatrix = MT)
  """

  if ~torch.cuda.is_available():
    device = "cpu"
    print("GPU not available")
	
  n = adjMatrix.shape[0]
  a = torch.ones([1,n]).to(device)
  h = torch.ones([1,n]).to(device)
  pa = a
  authority = {}
  hub = {}

  for k in range(1,p):
    h1 = torch.mm(adjMatrix, pa.T)/torch.norm(torch.mm(adjMatrix, pa.T))
    a1 = torch.mm(adjMatrix.T, h1)/torch.norm(torch.mm(adjMatrix.T , h1))

    h = torch.vstack((h,torch.mv(adjMatrix, a[k-1,:].T)/torch.norm(torch.mv(adjMatrix, a[k-1,:].T))))
    a = torch.vstack((a,torch.mv(adjMatrix.T, h[k,:].T)/torch.norm(torch.mv(adjMatrix.T, h[k,:].T))))

    pa = a1.T

    for i in range(n):
      authority[str(i)] = a[-1,i]
      hub[str(i)] = h[-1,i]
  return hub, authority, h, a