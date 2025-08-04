"""
Created on Fri Jan  28 09:56:07 2022
Last update on Sun Aug 03 15:02:36 2025

@author: nico
"""

import torch

def safe_normalize(vec: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vec)
    return vec / norm if norm > 0 else torch.zeros_like(vec)

def hits(adjMatrix: torch.Tensor, p: int = 100, device=0):
    """
    Compute HITS (Hub and Authority) scores on a graph represented by an adjacency matrix.

    Parameters
    ----------
    adjMatrix : torch.Tensor
        Square adjacency matrix of shape (N, N).
    p : int, optional
        Number of iterations to perform (default is 100).
    device : int or str, optional
        GPU device ID or "cpu". If GPU is unavailable, it defaults to "cpu".

    Returns
    -------
    hub : dict
        Final hub scores as a dictionary of node index -> score.
    authority : dict
        Final authority scores as a dictionary of node index -> score.
    h_all : torch.Tensor
        Hub scores over iterations, shape (p, N).
    a_all : torch.Tensor
        Authority scores over iterations, shape (p, N).
    """

    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    device = torch.device(device)
    n = adjMatrix.shape[0]

    # Initialize scores (iteration 0)
    h_all = torch.ones((1, n), device=device)
    a_all = torch.ones((1, n), device=device)

    # Iterative computation
    for _ in range(1, p):
        prev_a = a_all[-1]
        h_new = safe_normalize(torch.mv(adjMatrix, prev_a))
        h_all = torch.vstack((h_all, h_new.unsqueeze(0)))

        a_new = safe_normalize(torch.mv(adjMatrix.T, h_new))
        a_all = torch.vstack((a_all, a_new.unsqueeze(0)))

    # Final hub and authority scores (last iteration)
    hub = {str(i): h_all[-1, i].item() for i in range(n)}
    authority = {str(i): a_all[-1, i].item() for i in range(n)}

    return hub, authority, h_all, a_all

import torch

def tophits(T: torch.Tensor, epsilon: float = 1e-3, max_iter: int = 1000, device=0):
    """
    Compute TOPHITS scores for a 3D tensor using iterative decomposition.

    Parameters
    ----------
    T : torch.Tensor
        A 3D tensor of shape (N, M, Q).
    epsilon : float, optional
        Convergence threshold on lambda difference (default is 1e-3).
    max_iter : int, optional
        Maximum number of iterations to prevent infinite loops (default is 1000).
    device : int or str, optional
        GPU device ID or "cpu". If GPU is unavailable, it defaults to "cpu".

    Returns
    -------
    u : torch.Tensor
        Score vector for the first dimension (users), shape (N,).
    v : torch.Tensor
        Score vector for the second dimension (items), shape (M,).
    w : torch.Tensor
        Score vector for the third dimension (keywords), shape (Q,).
    """

    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    device = torch.device(device)

    N, M, Q = T.shape

    # Initialize vectors
    u = torch.ones(N, device=device)
    v = torch.ones(M, device=device)
    w = torch.ones(Q, device=device)

    u = u / torch.linalg.norm(u)
    v = v / torch.linalg.norm(v)
    w = w / torch.linalg.norm(w)

    lambda_prev = 0.0

    for _ in range(max_iter):
        # Update u
        uv = torch.tensordot(T, v, dims=([1], [0]))    # shape: (N, Q)
        u_new = torch.tensordot(uv, w, dims=([1], [0]))  # shape: (N,)
        u_new = u_new / torch.linalg.norm(u_new)

        # Update v
        vt = torch.tensordot(T, u_new, dims=([0], [0]))  # shape: (M, Q)
        v_new = torch.tensordot(vt, w, dims=([1], [0]))  # shape: (M,)
        v_new = v_new / torch.linalg.norm(v_new)

        # Update w
        wt = torch.tensordot(T, u_new, dims=([0], [0]))  # shape: (M, Q)
        w_new = torch.tensordot(wt, v_new, dims=([0], [0]))  # shape: (Q,)
        w_new = w_new / torch.linalg.norm(w_new)

        # Check convergence
        lambda_curr = (
            torch.linalg.norm(u_new) *
            torch.linalg.norm(v_new) *
            torch.linalg.norm(w_new)
        ).item()

        if abs(lambda_curr - lambda_prev) < epsilon:
            break

        lambda_prev = lambda_curr
        u, v, w = u_new, v_new, w_new

    return u, v, w


def socialAU(mu, mi, mw, T, epsilon: float = 0.001, device = 0):
    """
    Calculate the socialAU score in a 3 layer net and detect the influencer. 
    Corrected implementation following the paper's pseudocode exactly.

    Parameters
    -------------------
    mu, mi, mw: torch tensor. These are 3 adjiacency matrix with dimension NxN, MxM, QxQ.
    T: torch tensor. A 3D tensor with dimension NxMxQ.
    epsilon: float. Default 0.001. Stop criteria. If the labda value converge (|lambda(t-1) - labda(t)| < epsilon) stop the execution.
    device: Default 0. Set the GPU. If GPU is not available it is set to "cpu" automatically.

    Returns
    -------------------
    u, v, w: torch tensor. The socialAU scores for each node about, respectively, the first, second and third dimension of tensor (3 social networks).
    """
    
    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    # Initialize vectors to unit vectors (step 1 of pseudocode)
    n, m, r = T.shape[0], T.shape[1], T.shape[2]
    
    # For users layer
    a_U = torch.ones(n).to(device)
    h_U = torch.ones(n).to(device)
    
    # For items layer  
    a_I = torch.ones(m).to(device)
    h_I = torch.ones(m).to(device)
    
    # For keywords layer
    a_k = torch.ones(r).to(device)
    h_k = torch.ones(r).to(device)
    
    # For tensor decomposition
    h = torch.ones(n).to(device)  # users (first dimension)
    a = torch.ones(m).to(device)  # items (second dimension)  
    w = torch.ones(r).to(device)  # keywords (third dimension)
    
    # Initialize lambda (step 2)
    lambda_prev = 0
    
    # Main iteration loop (step 3)
    continua = True
    
    while continua:
        # Step 4-5: HITS for users layer
        h_U = torch.mv(mu, a_U)
        a_U = torch.mv(mu.T, h_U)
        
        # Step 6-7: HITS for items layer  
        h_I = torch.mv(mi, a_I)
        a_I = torch.mv(mi.T, h_I)
        
        # Step 8-9: HITS for keywords layer
        h_k = torch.mv(mw, a_k) 
        a_k = torch.mv(mw.T, h_k)
        
        # Step 10: Update h using tensor operations + HITS scores
        # h^(t+1) = A ×₂ a^(t) ×₃ w^(t) + h_U^(t+1) + a_U^(t+1)
        tensor_part = torch.tensordot(T, a, dims=([1], [0]))  # Contract dimension 1 with items
        tensor_part = torch.tensordot(tensor_part, w, dims=([1], [0]))  # Contract dimension 1 with keywords
        h = tensor_part + h_U + a_U
        
        # Step 11: Update a using tensor operations
        # a^(t+1) = A ×₁ h^(t+1) ×₃ w^(t)
        tensor_part = torch.tensordot(T, h, dims=([0], [0]))  # Contract dimension 0 with users
        a = torch.tensordot(tensor_part, w, dims=([1], [0]))  # Contract dimension 1 with keywords
        
        # Step 12: Update w using tensor operations + HITS scores
        # w^(t+1) = A ×₁ h^(t+1) ×₂ a^(t) + a_k^(t+1)
        tensor_part = torch.tensordot(T, h, dims=([0], [0]))  # Contract dimension 0 with users
        tensor_part = torch.tensordot(tensor_part, a, dims=([0], [0]))  # Contract dimension 0 with items
        w = tensor_part + a_k
        
        # Step 13: Calculate lambda = ||h|| ||a|| ||w||
        lambda_current = torch.linalg.norm(h) * torch.linalg.norm(a) * torch.linalg.norm(w)
        
        # Step 14: Normalize all vectors
        h = h / torch.linalg.norm(h)
        a = a / torch.linalg.norm(a) 
        w = w / torch.linalg.norm(w)
        
        # Normalize HITS vectors for next iteration
        a_U = a_U / torch.linalg.norm(a_U)
        h_U = h_U / torch.linalg.norm(h_U)
        a_I = a_I / torch.linalg.norm(a_I)
        h_I = h_I / torch.linalg.norm(h_I)
        a_k = a_k / torch.linalg.norm(a_k)
        h_k = h_k / torch.linalg.norm(h_k)
        
        # Step 15-16: Check convergence
        if abs(lambda_current - lambda_prev) <= epsilon:
            continua = False
        else:
            lambda_prev = lambda_current
    
    # Step 19: Return results
    # Reshape to match original function's output format
    u = h.unsqueeze(0)  # users scores
    v = a.unsqueeze(0)  # items scores  
    w_out = w.unsqueeze(0)  # keywords scores
    
    return u, v, w_out