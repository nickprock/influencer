"""
Created on Fri Jan  28 09:56:07 2022
Last update on Sat Feb 12 15:30:22 2022

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

  if not torch.cuda.is_available():
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

def tophits(T, epsilon: float = 0.001, device = 0):
    """
    Calculate the TOPHITS score for 3D tensor.

    Parameters
	  -------------------
	  T: a tensor NxMxQ.
	  epsilon: float. Default 0.001. Stop criteria. If the labda value converge (|lambda(t-1) - labda(t)| < epsilon) stop the execution.

	  Returns
	  -------------------
	  u, v, w: tensors. The TOPHITS scores for each node about, respectively, the first, second and third dimension of tensor T.

	  Example
	  -------------------
	  >> import numpy as np
	  >> from influencer.centrality import tophits

	  Create a 3D tensor in numpy

	  >> ten = np.random.rand(10,5,20)
	  >> ten[ten>0.5]=1
	  >> ten[ten<=0.5]=0
    >> MT = torch.from_numpy(ten).float().to(0) 
    >> # .to(0) if GPU is available, test your GPU with torch.cuda.is_available()
	  >> u, v, w = tophits(T=MT)
    """

    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    u, v, w = torch.empty([1,T.shape[0]]).to(device), torch.empty([1,T.shape[1]]).to(device), torch.empty([1,T.shape[2]]).to(device)
    sigma = []

    x = torch.ones((T.shape[0], 1)).to(device)
    y = torch.ones((T.shape[1], 1)).to(device)
    z = torch.ones((T.shape[2], 1)).to(device)
    
    tx = torch.squeeze(x)
    ty = torch.squeeze(y)
    tz = torch.squeeze(z)

    lambda0=100
    continua=True
    num=1

    while(continua):
        x1 = torch.tensordot(T,ty, dims= ([1],[0]))
        x = torch.squeeze(torch.tensordot(x1,tz,dims=([1],[0])))
        
        y1 = torch.tensordot(T,x,dims=([0],[0]))
        y = torch.squeeze(torch.tensordot(y1,tz,dims=([1], [0])))
        
        z1 = torch.tensordot(T,x,dims=([0],[0]))
        z = torch.squeeze(torch.tensordot(z1,y,dims=([0],[0])))
        
        tx=x/torch.linalg.norm(x)
        ty=y/torch.linalg.norm(y)
        tz=z/torch.linalg.norm(z)
        
        lambda1 = (torch.linalg.norm(tx)*torch.linalg.norm(ty)*torch.linalg.norm(tz)).item()
        
        if(abs(lambda1-lambda0) < epsilon):
            continua=False
        
        lambda0 = lambda1
        num = num+1
        
    
    u = torch.vstack((u,tx))
    v = torch.vstack((v,ty))
    w = torch.vstack((w,tz))
    
    sigma.append(lambda1)
    
    u = u[1:,:]
    v = v[1:,:]
    w = w[1:,:]


    return u, v, w