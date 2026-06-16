import torch
import influencer
from influencer import edge2adj, hits, tophits, socialAU

torch.manual_seed(42)

# Print the installed package version
print("influencer version:", influencer.__version__)

# Build a directed weighted graph with 6 nodes and 10 edges
edges = [[0,1],[0,2],[1,2],[1,3],[2,3],[3,4],[4,5],[5,3],[2,5],[0,4]]
adj_np = edge2adj(edgelist=edges, dim=6)
adj = torch.tensor(adj_np, dtype=torch.float32)

# Compute HITS hub and authority scores over 20 iterations
hub, authority, _, _ = hits(adj, p=20)

# Print the top-3 hub and authority nodes ranked by score
top_hubs = sorted(hub, key=hub.get, reverse=True)[:3]
top_auths = sorted(authority, key=authority.get, reverse=True)[:3]
print("Top-3 hubs:      ", top_hubs)
print("Top-3 authorities:", top_auths)

# Build a random binary 4x3x5 tensor and run TOPHITS decomposition
T = torch.randint(0, 2, (4, 3, 5)).float()
u, v, w = tophits(T)

# Print the highest-scoring node index along each tensor dimension
print("Top node dim 0 (users)   :", torch.argmax(u).item())
print("Top node dim 1 (items)   :", torch.argmax(v).item())
print("Top node dim 2 (keywords):", torch.argmax(w).item())
