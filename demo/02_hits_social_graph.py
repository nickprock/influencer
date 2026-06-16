import torch
from influencer import edge2adj, hits

# Edge list: [src, dst] means "src retweets dst" (src endorses dst's content)
# 1,2,3 retweet the media outlet (0); 1 and 2 cross-retweet; 4,5,6 only retweet 0; 7 retweets 1
edges = [
    [1, 0], [2, 0], [3, 0],   # users 1,2,3 retweet media outlet 0
    [1, 2], [2, 1],            # users 1 and 2 retweet each other
    [4, 0], [5, 0], [6, 0],   # users 4,5,6 retweet media outlet 0
    [7, 1],                    # user 7 retweets user 1
]

# Build adjacency matrix and convert to float tensor
adj_np = edge2adj(edgelist=edges, dim=8)
adj = torch.tensor(adj_np, dtype=torch.float32)

# Run HITS for 50 iterations
hub, authority, _, _ = hits(adj, p=50)

# --- Print authority table (sorted descending) ---
auth_ranked = sorted(authority.items(), key=lambda x: x[1], reverse=True)
hub_lookup  = hub

print("Ranked by Authority score")
print(f"{'Rank':>4} | {'User':>4} | {'Authority':>9} | {'Hub':>9}")
print(f"{'----':>4} | {'----':>4} | {'---------':>9} | {'---------':>9}")
for rank, (user, auth_score) in enumerate(auth_ranked, start=1):
    print(f"{rank:>4} | {user:>4} | {auth_score:>9.4f} | {hub_lookup[user]:>9.4f}")

print()

# --- Print hub table (sorted descending) ---
hub_ranked = sorted(hub.items(), key=lambda x: x[1], reverse=True)
auth_lookup = authority

print("Ranked by Hub score")
print(f"{'Rank':>4} | {'User':>4} | {'Hub':>9} | {'Authority':>9}")
print(f"{'----':>4} | {'----':>4} | {'---------':>9} | {'---------':>9}")
for rank, (user, hub_score) in enumerate(hub_ranked, start=1):
    print(f"{rank:>4} | {user:>4} | {hub_score:>9.4f} | {auth_lookup[user]:>9.4f}")

print()

# Validate expected structural outcomes
top_authority = max(authority, key=authority.get)
top_hub       = max(hub, key=hub.get)
assert top_authority == '0', f"Expected user 0 as top authority, got {top_authority}"
assert top_hub in ('1', '2'), f"Expected user 1 or 2 as top hub, got {top_hub}"

print(
    "Interpretation: In this Twitter-like network, authority scores capture how much a user\n"
    "is trusted as a source of information — user 0 (the media outlet) scores highest because\n"
    "six different users retweet it. Hub scores capture how well a user curates quality content\n"
    "by pointing to authoritative sources: users 1 and 2 score highest because they retweet\n"
    "the media outlet AND each other, forming a mutually reinforcing endorsement loop.\n"
    "Users 4-6 are pure consumers (high out-degree to one authority) so their hub scores are\n"
    "lower; user 7 only retweets user 1, whose authority is moderate, giving it the lowest hub."
)
