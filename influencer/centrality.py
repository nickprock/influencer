"""
Created on Fri Nov  29 20:02:07 2019

Last update on Sun Aug 03 15:57:28 2025


@author: nico
"""
import numpy as np

def safe_normalize(vec):
    """Safe normalization to avoid division by zero"""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.zeros_like(vec)

def hits(adjMatrix, p: int = 100):
    """
    Calculate the hub and authority score in a net

    Parameters
    -------------------
    adjMatrix: a numpy array NxN.

    p: int. Default 100. The max iteration.

    Returns
    -------------------
    hub: dict. The hub score for each node in the net.
    authority: dict. The authority score for each node in the net.
    h: numpy array pxN. The hub score for each node in the net for each algorithm's step.
    a: numpy array pxN. The authority score for each node in the net for each algorithm's step.

    Example
    -------------------
    >> import numpy as np
    >> from influencer.centrality import hits

    Create an adjiacency matrix with numpy

    >> adjM = np.random.rand(10, 10)
    >> adjM[adjM>0.5]=1
    >> adjM[adjM<=0.5]=0
    >> hub, aut, _, _ = hits(adjMatrix = adjM)
    """

    n = adjMatrix.shape[0]
    
    # Initialize with ones and normalize
    h_all = np.ones((1, n))
    a_all = np.ones((1, n))
    
    authority = {}
    hub = {}
    
    # Iterative computation
    for _ in range(1, p):
        prev_a = a_all[-1]
        h_new = safe_normalize(np.dot(adjMatrix, prev_a))
        h_all = np.vstack((h_all, h_new.reshape(1, -1)))
        
        a_new = safe_normalize(np.dot(adjMatrix.T, h_new))
        a_all = np.vstack((a_all, a_new.reshape(1, -1)))
    
    # Final hub and authority scores (last iteration)
    for i in range(n):
        authority[str(i)] = a_all[-1, i]
        hub[str(i)] = h_all[-1, i]
    
    return hub, authority, h_all, a_all


def tophits(T, epsilon: float = 0.001, max_iter: int = 1000):
    """
    Calculate the TOPHITS score for 3D tensor.

    Parameters
    -------------------
    T: a numpy array NxMxQ.
    epsilon: float. Default 0.001. Stop criteria. If the labda value converge (|lambda(t-1) - labda(t)| < epsilon) stop the execution.
    max_iter: int. Default 1000. Maximum number of iterations to prevent infinite loops.

    Returns
    -------------------
    u, v, w: numpy array. The TOPHITS scores for each node about, respectively, the first, second and third dimension of tensor T.

    Example
    -------------------
    >> import numpy as np
    >> from influencer.centrality import tophits

    Create a 3D tensor in numpy

    >> ten = np.random.rand(10,5,20)
    >> ten[ten>0.5]=1
    >> ten[ten<=0.5]=0
    >> d1, d2, d3 = tophits(T=ten)
    """

    N, M, Q = T.shape

    # Initialize vectors and normalize
    u = np.ones(N)
    v = np.ones(M)
    w = np.ones(Q)
    
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = w / np.linalg.norm(w)
    
    lambda_prev = 0.0
    
    for _ in range(max_iter):
        # Update u
        uv = np.tensordot(T, v, axes=([1], [0]))    # shape: (N, Q)
        u_new = np.tensordot(uv, w, axes=([1], [0]))  # shape: (N,)
        u_new = u_new / np.linalg.norm(u_new)
        
        # Update v
        vt = np.tensordot(T, u_new, axes=([0], [0]))  # shape: (M, Q)
        v_new = np.tensordot(vt, w, axes=([1], [0]))  # shape: (M,)
        v_new = v_new / np.linalg.norm(v_new)
        
        # Update w
        wt = np.tensordot(T, u_new, axes=([0], [0]))  # shape: (M, Q)
        w_new = np.tensordot(wt, v_new, axes=([0], [0]))  # shape: (Q,)
        w_new = w_new / np.linalg.norm(w_new)
        
        # Check convergence
        lambda_curr = (
            np.linalg.norm(u_new) *
            np.linalg.norm(v_new) *
            np.linalg.norm(w_new)
        )
        
        if abs(lambda_curr - lambda_prev) < epsilon:
            break
            
        lambda_prev = lambda_curr
        u, v, w = u_new, v_new, w_new

    return u.reshape(1, -1), v.reshape(1, -1), w.reshape(1, -1)


def socialAU(mu, mi, mw, T, epsilon: float = 0.001):
    """
    Calculate the socialAU score in a 3 layer net and detect the influencer. 
    Corrected implementation following the paper's pseudocode exactly.

    Parameters
    -------------------
    mu, mi, mw: numpy array. These are 3 adjiacency matrix with dimension NxN, MxM, QxQ.
    T: numpy array. A 3D tensor with dimension NxMxQ.
    epsilon: float. Default 0.001. Stop criteria. If the labda value converge (|lambda(t-1) - labda(t)| < epsilon) stop the execution.
    
    Returns
    -------------------
    u, v, w: numpy array. The socialAU scores for each node about, respectively, the first, second and third dimension of tensor (3 social networks).

    Example
    -------------------
    >> import numpy as np
    >> from influencer.centrality import socialAU

    Create 3 adjiacency matrix
    >> userNet = np.random.rand(10, 10)
    >> userNet[userNet>0.5]=1
    >> userNet[userNet<=0.5]=0
    >> itemNet = np.random.rand(5, 5)
    >> itemNet[itemNet>0.5]=1
    >> itemNet[itemNet<=0.5]=0
    >> wordNet = np.random.rand(20, 20)
    >> wordNet[wordNet>0.5]=1
    >> wordNet[wordNet<=0.5]=0

    Create a 3D tensor in numpy

    >> ten = np.random.rand(10,5,20)
    >> ten[ten>0.5]=1
    >> ten[ten<=0.5]=0

    >> user, item, word = socialAU(userNet, itemNet, wordNet,ten)
    """

    # Initialize vectors to unit vectors (step 1 of pseudocode)
    n, m, r = T.shape[0], T.shape[1], T.shape[2]
    
    # For users layer
    a_U = np.ones(n)
    h_U = np.ones(n)
    
    # For items layer  
    a_I = np.ones(m)
    h_I = np.ones(m)
    
    # For keywords layer
    a_k = np.ones(r)
    h_k = np.ones(r)
    
    # For tensor decomposition
    h = np.ones(n)  # users (first dimension)
    a = np.ones(m)  # items (second dimension)  
    w = np.ones(r)  # keywords (third dimension)
    
    # Initialize lambda (step 2)
    lambda_prev = 0
    
    # Main iteration loop (step 3)
    continua = True
    
    while continua:
        # Step 4-5: HITS for users layer
        h_U = np.dot(mu, a_U)
        a_U = np.dot(mu.T, h_U)
        
        # Step 6-7: HITS for items layer  
        h_I = np.dot(mi, a_I)
        a_I = np.dot(mi.T, h_I)
        
        # Step 8-9: HITS for keywords layer
        h_k = np.dot(mw, a_k) 
        a_k = np.dot(mw.T, h_k)
        
        # Step 10: Update h using tensor operations + HITS scores
        # h^(t+1) = A ×₂ a^(t) ×₃ w^(t) + h_U^(t+1) + a_U^(t+1)
        tensor_part = np.tensordot(T, a, axes=([1], [0]))  # Contract dimension 1 with items
        tensor_part = np.tensordot(tensor_part, w, axes=([1], [0]))  # Contract dimension 1 with keywords
        h = tensor_part + h_U + a_U
        
        # Step 11: Update a using tensor operations
        # a^(t+1) = A ×₁ h^(t+1) ×₃ w^(t)
        tensor_part = np.tensordot(T, h, axes=([0], [0]))  # Contract dimension 0 with users
        a = np.tensordot(tensor_part, w, axes=([1], [0]))  # Contract dimension 1 with keywords
        
        # Step 12: Update w using tensor operations + HITS scores
        # w^(t+1) = A ×₁ h^(t+1) ×₂ a^(t) + a_k^(t+1)
        tensor_part = np.tensordot(T, h, axes=([0], [0]))  # Contract dimension 0 with users
        tensor_part = np.tensordot(tensor_part, a, axes=([0], [0]))  # Contract dimension 0 with items
        w = tensor_part + a_k
        
        # Step 13: Calculate lambda = ||h|| ||a|| ||w||
        lambda_current = np.linalg.norm(h) * np.linalg.norm(a) * np.linalg.norm(w)
        
        # Step 14: Normalize all vectors
        h = h / np.linalg.norm(h)
        a = a / np.linalg.norm(a) 
        w = w / np.linalg.norm(w)
        
        # Normalize HITS vectors for next iteration
        a_U = a_U / np.linalg.norm(a_U)
        h_U = h_U / np.linalg.norm(h_U)
        a_I = a_I / np.linalg.norm(a_I)
        h_I = h_I / np.linalg.norm(h_I)
        a_k = a_k / np.linalg.norm(a_k)
        h_k = h_k / np.linalg.norm(h_k)
        
        # Step 15-16: Check convergence
        if abs(lambda_current - lambda_prev) <= epsilon:
            continua = False
        else:
            lambda_prev = lambda_current
    
    # Step 19: Return results
    # Reshape to match original function's output format
    u = h.reshape(1, -1)  # users scores
    v = a.reshape(1, -1)  # items scores  
    w_out = w.reshape(1, -1)  # keywords scores
    
    return u, v, w_out