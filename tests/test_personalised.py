"""
Test suite for influencer/personalised.py
"""

import pytest
import torch
from unittest.mock import patch

from influencer.personalised import socialAU_personalised, get_ego_network_seeds
from influencer.torch_centrality import socialAU


# ---------------------------------------------------------------------------
# socialAU_personalised
# ---------------------------------------------------------------------------

class TestSocialAUPersonalised:

    @pytest.fixture
    def sample_inputs(self):
        torch.manual_seed(0)
        mu = torch.rand(4, 4).abs()
        mi = torch.rand(3, 3).abs()
        mw = torch.rand(5, 5).abs()
        T = torch.rand(4, 3, 5)
        T[T < 0.7] = 0.0
        return mu, mi, mw, T

    @patch('torch.cuda.is_available', return_value=False)
    def test_full_teleportation_top_ranks_seed_user(self, mock_cuda):
        """alpha=1.0 ignores graph structure entirely on a_U: each iteration
        a_U becomes exactly the one-hot seed vector, so with T=0 (no
        confounding content signal) the seed user must dominate h."""
        # 1 -> 3, 2 -> 3: node 3 is the *natural* authority without
        # personalisation (everyone points to it, nobody points to node 0).
        mu = torch.zeros(4, 4)
        mu[1, 3] = 1.0
        mu[2, 3] = 1.0
        mi = torch.eye(3)
        mw = torch.eye(5)
        T = torch.zeros(4, 3, 5)

        u, v, w = socialAU_personalised(
            mu, mi, mw, T, seed_users=[0], alpha=1.0, epsilon=1e-6,
        )

        assert torch.argmax(u[0]).item() == 0

    @patch('torch.cuda.is_available', return_value=False)
    def test_alpha_zero_matches_standard_socialau(self, mock_cuda, sample_inputs):
        mu, mi, mw, T = sample_inputs

        u1, v1, w1 = socialAU(mu, mi, mw, T, epsilon=1e-6)
        u2, v2, w2 = socialAU_personalised(
            mu, mi, mw, T, seed_users=[0, 1], alpha=0.0, epsilon=1e-6,
        )

        assert torch.allclose(u1, u2)
        assert torch.allclose(v1, v2)
        assert torch.allclose(w1, w2)

    @patch('torch.cuda.is_available', return_value=False)
    def test_personalised_differs_from_global_for_subset_seed(self, mock_cuda, sample_inputs):
        mu, mi, mw, T = sample_inputs

        u_global, _, _ = socialAU(mu, mi, mw, T, epsilon=1e-6)
        u_personal, _, _ = socialAU_personalised(
            mu, mi, mw, T, seed_users=[0], alpha=0.15, epsilon=1e-6,
        )

        assert not torch.allclose(u_global, u_personal)

    @patch('torch.cuda.is_available', return_value=False)
    def test_output_shapes(self, mock_cuda, sample_inputs):
        mu, mi, mw, T = sample_inputs
        u, v, w = socialAU_personalised(
            mu, mi, mw, T, seed_users=[0], alpha=0.15, epsilon=0.01,
        )
        assert u.shape == (1, 4)
        assert v.shape == (1, 3)
        assert w.shape == (1, 5)

    @patch('torch.cuda.is_available', return_value=False)
    def test_output_normalized(self, mock_cuda, sample_inputs):
        mu, mi, mw, T = sample_inputs
        u, v, w = socialAU_personalised(
            mu, mi, mw, T, seed_users=[0], alpha=0.15, epsilon=0.01,
        )
        assert abs(torch.linalg.norm(u).item() - 1.0) < 1e-5
        assert abs(torch.linalg.norm(v).item() - 1.0) < 1e-5
        assert abs(torch.linalg.norm(w).item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# get_ego_network_seeds
# ---------------------------------------------------------------------------

class TestGetEgoNetworkSeeds:

    @pytest.fixture
    def chain_mu(self):
        # 0 -> 1 -> 2 -> 3
        adj = torch.zeros(4, 4)
        adj[0, 1] = 1.0
        adj[1, 2] = 1.0
        adj[2, 3] = 1.0
        return adj

    def test_hops_one_returns_direct_neighbours_only(self, chain_mu):
        seeds = get_ego_network_seeds(chain_mu, query_user=0, hops=1)
        assert sorted(seeds) == [0, 1]

    def test_hops_two_includes_two_hop_neighbours(self, chain_mu):
        seeds = get_ego_network_seeds(chain_mu, query_user=0, hops=2)
        assert sorted(seeds) == [0, 1, 2]

    def test_includes_query_user_itself(self, chain_mu):
        seeds = get_ego_network_seeds(chain_mu, query_user=3, hops=1)
        assert 3 in seeds

    def test_isolated_node_returns_only_itself(self):
        adj = torch.zeros(3, 3)
        seeds = get_ego_network_seeds(adj, query_user=1, hops=2)
        assert seeds == [1]

    def test_branching_graph_includes_all_direct_neighbours(self):
        # 0 -> 1, 0 -> 2
        adj = torch.zeros(3, 3)
        adj[0, 1] = 1.0
        adj[0, 2] = 1.0
        seeds = get_ego_network_seeds(adj, query_user=0, hops=1)
        assert sorted(seeds) == [0, 1, 2]

    def test_hops_exceeding_graph_diameter_stops_growing(self, chain_mu):
        seeds_2 = get_ego_network_seeds(chain_mu, query_user=0, hops=10)
        assert sorted(seeds_2) == [0, 1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
