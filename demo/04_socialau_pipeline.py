import torch
from influencer import edge2adj, socialAU

# Build the user adjacency matrix MU (5x5): users 1,2,3 follow user 0; user 3 also follows user 1
mu_edges = [[1,0],[2,0],[3,0],[3,1]]
MU = torch.tensor(edge2adj(edgelist=mu_edges, dim=5), dtype=torch.float32)

# Build the item adjacency matrix MI (4x4): products 0-1 and 2-3 form two similarity clusters
mi_edges = [[0,1],[1,0],[2,3],[3,2]]
MI = torch.tensor(edge2adj(edgelist=mi_edges, dim=4), dtype=torch.float32)

# Build the keyword co-occurrence matrix MK (6x6): keywords 0-2 and 3-5 form two clusters
mk_edges = [[0,1],[1,0],[0,2],[2,0],[1,2],[2,1],[3,4],[4,3],[3,5],[5,3],[4,5],[5,4]]
MK = torch.tensor(edge2adj(edgelist=mk_edges, dim=6), dtype=torch.float32)

# Build the interaction tensor T (5x4x6): user 0 heavily reviews product 0 with keywords 0,1,2
T = torch.zeros(5, 4, 6)
T[0,0,0] = 5; T[0,0,1] = 4; T[0,0,2] = 2  # dominant: user 0, product 0, keywords 0-2
T[1,1,3] = 1                                 # user 1, product 1, keyword 3
T[2,2,3] = 1                                 # user 2, product 2, keyword 3
T[3,0,0] = 1                                 # user 3, product 0, keyword 0
T[4,3,5] = 1                                 # user 4, product 3, keyword 5

# Run the full SocialAU pipeline
h_raw, a_raw, w_raw = socialAU(MU, MI, MK, T, epsilon=1e-6)
h = h_raw.squeeze(0)  # (5,)  user scores
a = a_raw.squeeze(0)  # (4,)  item scores
w = w_raw.squeeze(0)  # (6,)  keyword scores


def print_ranked(label, scores):
    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
    print(f"Ranked {label}:")
    print(f"  {'Rank':>4} | {'Index':>5} | {'Score':>12}")
    print(f"  {'----':>4} | {'-----':>5} | {'------------':>12}")
    for rank, (idx, score) in enumerate(ranked, start=1):
        print(f"  {rank:>4} | {idx:>5} | {score:>12.6f}")
    print()


# Print ranked tables for users, items, and keywords
print_ranked("Users     (h scores)", h)
print_ranked("Products  (a scores)", a)
print_ranked("Keywords  (w scores)", w)

# Validate that user 0 is the most authoritative
top_user = torch.argmax(h).item()
assert top_user == 0, f"Expected user 0 as top influencer, got user {top_user}"

top_item  = torch.argmax(a).item()
top_kws   = torch.topk(w, 2).indices.tolist()

print(
    "Interpretation\n"
    "--------------\n"
    f"  Most authoritative user : user {top_user}\n"
    f"  Top product reviewed    : product {top_item}\n"
    f"  Dominant keywords       : keyword {top_kws[0]}, keyword {top_kws[1]}\n"
    "\n"
    "  User 0 emerges as the platform's leading influencer because the socialAU algorithm\n"
    "  jointly propagates three signals: (1) in the user graph, three followers endorse\n"
    "  user 0, giving it the highest HITS authority among users; (2) user 0 concentrates\n"
    "  its reviews on product 0 with high interaction counts, pulling product 0 to the top\n"
    "  of the item ranking; (3) keywords 0 and 1 appear most in those reviews and form a\n"
    "  dense co-occurrence cluster, amplifying their keyword scores. The tensor T couples\n"
    "  all three layers so that a trusted user reviewing a well-connected product with\n"
    "  cohesive keywords reinforces every score simultaneously."
)
