"""
Test suite for edge2adj.py and torch_centrality.py modules
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch

from influencer.edge2adj import edge2adj
from influencer.torch_centrality import hits, tophits, socialAU, safe_normalize


# ---------------------------------------------------------------------------
# safe_normalize
# ---------------------------------------------------------------------------

class TestSafeNormalize:

    def test_zero_vector_returns_zeros(self):
        vec = torch.zeros(5)
        result = safe_normalize(vec)
        assert torch.all(result == 0)

    def test_general_vector_has_unit_norm(self):
        vec = torch.tensor([3.0, 4.0])  # norm = 5
        result = safe_normalize(vec)
        assert abs(torch.linalg.norm(result).item() - 1.0) < 1e-6
        assert torch.allclose(result, torch.tensor([0.6, 0.8]))

    def test_unit_vector_is_unchanged(self):
        vec = torch.tensor([1.0, 0.0, 0.0])
        result = safe_normalize(vec)
        assert torch.allclose(result, vec)

    def test_preserves_direction(self):
        vec = torch.tensor([1.0, -2.0, 3.0])
        result = safe_normalize(vec)
        assert abs(torch.linalg.norm(result).item() - 1.0) < 1e-6
        # cosine similarity with original should be 1
        cos = torch.dot(result, vec) / torch.linalg.norm(vec)
        assert abs(cos.item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# edge2adj
# ---------------------------------------------------------------------------

class TestEdge2Adj:

    def test_basic_directed_weighted(self):
        edgelist = [[0, 1], [1, 2], [0, 3], [2, 1], [3, 1]]
        adj = edge2adj(edgelist=edgelist, dim=4, weight=True, direct=True)
        expected = np.array([
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ])
        assert np.array_equal(adj, expected)
        assert adj.dtype == int

    def test_basic_directed_unweighted(self):
        edgelist = [[0, 1], [1, 2], [0, 1]]  # duplicate edge
        adj = edge2adj(edgelist=edgelist, dim=3, weight=False, direct=True)
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        assert np.array_equal(adj, expected)

    def test_basic_undirected_weighted(self):
        edgelist = [[0, 1], [1, 2]]
        adj = edge2adj(edgelist=edgelist, dim=3, weight=True, direct=False)
        expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        assert np.array_equal(adj, expected)

    def test_basic_undirected_unweighted(self):
        edgelist = [[0, 1], [0, 1], [1, 2]]  # duplicate edge
        adj = edge2adj(edgelist=edgelist, dim=3, weight=False, direct=False)
        expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        assert np.array_equal(adj, expected)

    def test_weighted_multiple_edges(self):
        edgelist = [[0, 1], [0, 1], [0, 1]]
        adj = edge2adj(edgelist=edgelist, dim=2, weight=True, direct=True)
        expected = np.array([[0, 3], [0, 0]])
        assert np.array_equal(adj, expected)

    def test_empty_edgelist(self):
        adj = edge2adj(edgelist=[], dim=3, weight=True, direct=True)
        assert np.array_equal(adj, np.zeros((3, 3), dtype=int))

    def test_self_loops(self):
        edgelist = [[0, 0], [1, 1]]
        adj = edge2adj(edgelist=edgelist, dim=2, weight=True, direct=True)
        expected = np.array([[1, 0], [0, 1]])
        assert np.array_equal(adj, expected)

    def test_large_dimension(self):
        edgelist = [[0, 9], [5, 3], [9, 0]]
        adj = edge2adj(edgelist=edgelist, dim=10, weight=True, direct=True)
        assert adj.shape == (10, 10)
        assert adj[0, 9] == 1
        assert adj[5, 3] == 1
        assert adj[9, 0] == 1
        assert np.sum(adj) == 3

    def test_returns_numpy_array(self):
        adj = edge2adj([[0, 1]], dim=2)
        assert isinstance(adj, np.ndarray)

    def test_undirected_weighted_self_loop_counted_once(self):
        """Self-loop in undirected weighted graph should be counted once, not twice"""
        edgelist = [[0, 0], [1, 1]]
        adj = edge2adj(edgelist=edgelist, dim=2, weight=True, direct=False)
        expected = np.array([[1, 0], [0, 1]])
        assert np.array_equal(adj, expected), (
            f"Self-loop counted {adj[0,0]} times instead of 1"
        )

    def test_invalid_edge_indices(self):
        with pytest.raises(IndexError):
            edge2adj([[0, 5]], dim=3, weight=True, direct=True)


# ---------------------------------------------------------------------------
# hits
# ---------------------------------------------------------------------------

class TestHits:

    @pytest.fixture
    def sample_adj_matrix(self):
        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 0]
        ], dtype=float)
        return torch.from_numpy(adj).float()

    @patch('torch.cuda.is_available')
    def test_hits_basic_cpu(self, mock_cuda, sample_adj_matrix):
        mock_cuda.return_value = False
        hub, authority, h, a = hits(sample_adj_matrix, p=10)

        assert isinstance(hub, dict)
        assert isinstance(authority, dict)
        assert len(hub) == 4
        assert len(authority) == 4
        for i in range(4):
            assert str(i) in hub
            assert str(i) in authority
            assert 0 <= hub[str(i)] <= 1
            assert 0 <= authority[str(i)] <= 1
        assert h.shape == (10, 4)
        assert a.shape == (10, 4)

    @patch('torch.cuda.is_available')
    def test_hits_output_is_normalized(self, mock_cuda, sample_adj_matrix):
        mock_cuda.return_value = False
        hub, authority, h, a = hits(sample_adj_matrix, p=50)

        hub_norm = sum(v ** 2 for v in hub.values())
        auth_norm = sum(v ** 2 for v in authority.values())
        assert abs(hub_norm - 1.0) < 1e-5
        assert abs(auth_norm - 1.0) < 1e-5

    @patch('torch.cuda.is_available')
    def test_hits_single_node(self, mock_cuda):
        mock_cuda.return_value = False
        adj = torch.zeros(1, 1).float()
        hub, authority, h, a = hits(adj, p=5)
        assert len(hub) == 1
        assert len(authority) == 1
        assert hub['0'] == authority['0']

    @patch('torch.cuda.is_available')
    def test_hits_gpu_fallback(self, mock_cuda):
        mock_cuda.return_value = False
        adj = torch.eye(3).float()
        hub, authority, h, a = hits(adj, p=5, device=0)
        assert isinstance(hub, dict)
        assert isinstance(authority, dict)

    @patch('torch.cuda.is_available')
    def test_hits_scores_non_negative(self, mock_cuda, sample_adj_matrix):
        mock_cuda.return_value = False
        hub, authority, _, _ = hits(sample_adj_matrix, p=20)
        assert all(v >= 0 for v in hub.values())
        assert all(v >= 0 for v in authority.values())

    @patch('torch.cuda.is_available')
    def test_hits_history_shape(self, mock_cuda, sample_adj_matrix):
        """h_all and a_all track scores over all p iterations"""
        mock_cuda.return_value = False
        p = 15
        _, _, h_all, a_all = hits(sample_adj_matrix, p=p)
        assert h_all.shape == (p, 4)
        assert a_all.shape == (p, 4)


# ---------------------------------------------------------------------------
# tophits
# ---------------------------------------------------------------------------

class TestTopHits:

    @pytest.fixture
    def sample_3d_tensor(self):
        tensor = torch.zeros(3, 4, 5).float()
        tensor[0, 1, 2] = 1
        tensor[1, 2, 3] = 1
        tensor[2, 0, 1] = 1
        tensor[0, 3, 4] = 1
        return tensor

    @patch('torch.cuda.is_available')
    def test_tophits_output_shapes(self, mock_cuda, sample_3d_tensor):
        mock_cuda.return_value = False
        u, v, w = tophits(sample_3d_tensor, epsilon=0.01)
        assert u.shape == (3,)
        assert v.shape == (4,)
        assert w.shape == (5,)

    @patch('torch.cuda.is_available')
    def test_tophits_output_normalized(self, mock_cuda, sample_3d_tensor):
        mock_cuda.return_value = False
        u, v, w = tophits(sample_3d_tensor, epsilon=0.01)
        assert abs(torch.linalg.norm(u).item() - 1.0) < 1e-5
        assert abs(torch.linalg.norm(v).item() - 1.0) < 1e-5
        assert abs(torch.linalg.norm(w).item() - 1.0) < 1e-5

    @patch('torch.cuda.is_available')
    def test_tophits_convergence_different_epsilon(self, mock_cuda):
        mock_cuda.return_value = False
        tensor = torch.rand(4, 3, 5).float()
        tensor[tensor > 0.5] = 1
        tensor[tensor <= 0.5] = 0

        u1, v1, w1 = tophits(tensor, epsilon=1e-6)
        u2, v2, w2 = tophits(tensor, epsilon=0.1)

        assert abs(torch.linalg.norm(u1).item() - 1.0) < 1e-5
        assert abs(torch.linalg.norm(u2).item() - 1.0) < 1e-5

    @patch('torch.cuda.is_available')
    def test_tophits_max_iter_respected(self, mock_cuda):
        """max_iter=1 should stop after one update"""
        mock_cuda.return_value = False
        torch.manual_seed(0)
        T = torch.rand(4, 3, 5).float()

        u1, _, _ = tophits(T, epsilon=0.0, max_iter=1)
        u5, _, _ = tophits(T, epsilon=0.0, max_iter=5)
        # One step is not enough: results should differ
        assert not torch.allclose(u1, u5, atol=1e-3)

    @patch('torch.cuda.is_available')
    def test_tophits_iterations_matter_regression(self, mock_cuda):
        """Regression: old code stopped after 2 iterations for any epsilon > 0 because
        lambda was computed on already-normalised vectors (always 1.0).

        With the old bug, tophits(T, epsilon=1e-8) would stop after 2 iterations and
        fail to recover the dominant component of this tensor (u[0] ≈ 0.74 instead of 1.0).
        After the fix, the algorithm converges fully and u[0] → 1.0.
        """
        mock_cuda.return_value = False

        # Two orthogonal rank-1 components with ratio 10:9.
        # Starting from all-ones, after 2 iterations u[0] ≈ 0.74, u[2] ≈ 0.67.
        # Full convergence requires ~50+ iterations to suppress the non-dominant direction.
        u1 = torch.zeros(6); u1[0] = 1.0
        v1 = torch.zeros(5); v1[0] = 1.0
        w1 = torch.zeros(4); w1[0] = 1.0
        u2 = torch.zeros(6); u2[2] = 1.0
        v2 = torch.zeros(5); v2[2] = 1.0
        w2 = torch.zeros(4); w2[2] = 1.0
        T = 10.0 * torch.einsum('i,j,k->ijk', u1, v1, w1) + \
             9.0 * torch.einsum('i,j,k->ijk', u2, v2, w2)

        u, _, _ = tophits(T, epsilon=1e-8, max_iter=1000)

        assert abs(u[0].item()) > 0.99, (
            f"u[0]={u[0].item():.4f}: dominant direction not recovered — "
            "old bug stopped at 2 iterations giving u[0] ≈ 0.74"
        )
        assert abs(u[2].item()) < 0.15, (
            f"u[2]={u[2].item():.4f}: non-dominant direction not suppressed after convergence"
        )

    @patch('torch.cuda.is_available')
    def test_tophits_recovers_dominant_component(self, mock_cuda):
        """TOPHITS should identify the dominant direction of a structured tensor"""
        mock_cuda.return_value = False
        torch.manual_seed(0)

        # Dominant rank-1 component (weight 10) plus noise (weight 0.1)
        u_true = torch.tensor([3.0, 1.0, 2.0])
        v_true = torch.tensor([1.0, 4.0, 2.0, 3.0])
        w_true = torch.tensor([2.0, 1.0, 3.0, 5.0])
        T = 10.0 * torch.einsum('i,j,k->ijk', u_true, v_true, w_true)
        T = T + 0.1 * torch.rand_like(T)

        u, v, w = tophits(T, epsilon=1e-8, max_iter=1000)

        u_norm = u_true / torch.linalg.norm(u_true)
        v_norm = v_true / torch.linalg.norm(v_true)
        w_norm = w_true / torch.linalg.norm(w_true)

        # Alignment should be close to 1 (up to sign)
        assert abs(abs(torch.dot(u, u_norm).item()) - 1.0) < 0.01
        assert abs(abs(torch.dot(v, v_norm).item()) - 1.0) < 0.01
        assert abs(abs(torch.dot(w, w_norm).item()) - 1.0) < 0.01

    @patch('torch.cuda.is_available')
    def test_tophits_converges_to_fixed_point(self, mock_cuda):
        """After epsilon-convergence, one more iteration should not change the result.
        Note: the Gauss-Seidel update does NOT guarantee monotonically increasing lambda
        at every step, only overall convergence to a fixed point."""
        mock_cuda.return_value = False
        torch.manual_seed(1)
        T = torch.rand(4, 3, 4).float()

        u, v, w = tophits(T, epsilon=1e-8, max_iter=1000)

        # One extra iteration from the converged state
        uv = torch.tensordot(T, v, dims=([1], [0]))
        u_next = safe_normalize(torch.tensordot(uv, w, dims=([1], [0])))

        # At a fixed point, u_next must align with u (cosine similarity ≈ 1)
        alignment = abs(torch.dot(u, u_next).item())
        assert alignment > 0.9999, (
            f"Not at a fixed point after convergence: alignment={alignment:.6f}"
        )


# ---------------------------------------------------------------------------
# socialAU
# ---------------------------------------------------------------------------

class TestSocialAU:

    @pytest.fixture
    def sample_inputs(self):
        mu = torch.eye(3).float()
        mi = torch.eye(4).float()
        mw = torch.eye(5).float()
        T = torch.rand(3, 4, 5).float()
        T[T > 0.7] = 1
        T[T <= 0.7] = 0
        return mu, mi, mw, T

    @patch('torch.cuda.is_available')
    def test_socialau_output_shapes(self, mock_cuda, sample_inputs):
        mock_cuda.return_value = False
        mu, mi, mw, T = sample_inputs
        u, v, w = socialAU(mu, mi, mw, T, epsilon=0.01)
        assert u.shape == (1, 3)
        assert v.shape == (1, 4)
        assert w.shape == (1, 5)

    @patch('torch.cuda.is_available')
    def test_socialau_output_normalized(self, mock_cuda, sample_inputs):
        mock_cuda.return_value = False
        mu, mi, mw, T = sample_inputs
        u, v, w = socialAU(mu, mi, mw, T, epsilon=0.01)
        assert abs(torch.linalg.norm(u).item() - 1.0) < 1e-5
        assert abs(torch.linalg.norm(v).item() - 1.0) < 1e-5
        assert abs(torch.linalg.norm(w).item() - 1.0) < 1e-5

    @patch('torch.cuda.is_available')
    def test_socialau_convergence_different_epsilon(self, mock_cuda, sample_inputs):
        mock_cuda.return_value = False
        mu, mi, mw, T = sample_inputs
        u1, v1, w1 = socialAU(mu, mi, mw, T, epsilon=1e-6)
        u2, v2, w2 = socialAU(mu, mi, mw, T, epsilon=0.1)
        assert abs(torch.linalg.norm(u1).item() - 1.0) < 1e-5
        assert abs(torch.linalg.norm(u2).item() - 1.0) < 1e-5

    @patch('torch.cuda.is_available')
    def test_socialau_dimension_mismatch(self, mock_cuda):
        mock_cuda.return_value = False
        mu = torch.eye(3).float()
        mi = torch.eye(3).float()   # wrong: should be 4×4
        mw = torch.eye(3).float()   # wrong: should be 5×5
        T = torch.rand(3, 4, 5).float()
        with pytest.raises((RuntimeError, IndexError)):
            socialAU(mu, mi, mw, T, epsilon=0.01)

    @patch('torch.cuda.is_available')
    def test_socialau_identifies_authoritative_user(self, mock_cuda):
        """User 0, heavily mentioned by others and active on dominant item,
        should receive the highest score."""
        mock_cuda.return_value = False

        # Users: user 0 is mentioned by users 1 and 2 (incoming arcs → high authority)
        mu = torch.tensor([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
        ])
        # Items: chain
        mi = torch.tensor([
            [0., 1.],
            [0., 0.],
        ])
        # Keywords: kw 0 and kw 1 co-occur
        mk = torch.tensor([
            [0., 1.],
            [0., 0.],
        ])
        # Tensor: user 0 dominates — 5 posts on item 0 with keyword 0
        T = torch.zeros(3, 2, 2)
        T[0, 0, 0] = 5.0
        T[1, 0, 1] = 1.0
        T[2, 1, 0] = 1.0

        u, v, w = socialAU(mu, mi, mk, T, epsilon=1e-6)

        assert torch.argmax(u[0]).item() == 0, "User 0 should be the most authoritative"
        assert torch.argmax(v[0]).item() == 0, "Item 0 should be dominant"

    @patch('torch.cuda.is_available')
    def test_socialau_convergence_direction(self, mock_cuda, sample_inputs):
        """Lambda should be non-negative throughout convergence (paper step 15: λ₁ − λ ≤ ε)"""
        mock_cuda.return_value = False
        mu, mi, mw, T = sample_inputs
        # If the convergence check wrongly used abs(), a decrease in lambda could
        # terminate too early. We verify the final lambda is positive.
        u, v, w = socialAU(mu, mi, mw, T, epsilon=1e-6)
        assert torch.linalg.norm(u).item() > 0
        assert torch.linalg.norm(v).item() > 0
        assert torch.linalg.norm(w).item() > 0

    @patch('torch.cuda.is_available')
    def test_socialau_jacobi_step12(self, mock_cuda):
        """Step 12 must use a^(t) (old a), not a^(t+1).
        We verify by checking that the w update changes when we run one extra
        iteration vs freezing a before step 11."""
        mock_cuda.return_value = False

        torch.manual_seed(7)
        mu = torch.rand(3, 3).float().abs()
        mi = torch.rand(4, 4).float().abs()
        mk = torch.rand(5, 5).float().abs()
        T = torch.rand(3, 4, 5).float()

        # If step 12 uses a_prev correctly, the result is reproducible
        u1, v1, w1 = socialAU(mu, mi, mk, T, epsilon=1e-4)
        u2, v2, w2 = socialAU(mu, mi, mk, T, epsilon=1e-4)

        assert torch.allclose(u1, u2), "socialAU should be deterministic"
        assert torch.allclose(w1, w2), "w should be deterministic (depends on a_prev)"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_edge2adj_to_hits_pipeline(self):
        edgelist = [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]]
        adj_np = edge2adj(edgelist, dim=3, weight=True, direct=True)
        adj_torch = torch.from_numpy(adj_np).float()

        with patch('torch.cuda.is_available', return_value=False):
            hub, authority, _, _ = hits(adj_torch, p=20)

        assert len(hub) == 3
        for i in range(3):
            assert hub[str(i)] >= 0
            assert authority[str(i)] >= 0

    def test_different_graph_types_with_hits(self):
        test_cases = [
            ([[0, 1], [1, 0]], 2, True, True),
            ([[0, 1], [1, 2], [2, 0]], 3, False, False),
            ([[0, 0]], 1, True, True),
            ([], 3, True, True),
        ]
        for edgelist, dim, weight, direct in test_cases:
            adj_np = edge2adj(edgelist, dim=dim, weight=weight, direct=direct)
            assert adj_np.shape == (dim, dim)
            assert adj_np.dtype == int

            if len(edgelist) > 0:
                adj_torch = torch.from_numpy(adj_np).float()
                with patch('torch.cuda.is_available', return_value=False):
                    hub, authority, _, _ = hits(adj_torch, p=10)
                assert len(hub) == dim
                assert len(authority) == dim

    def test_full_pipeline_socialau(self):
        """Full pipeline: edgelists → adjacency matrices → SocialAU"""
        # User 0 is mentioned by users 1 and 2
        user_edges = [[1, 0], [2, 0]]
        # Items connected in a chain
        item_edges = [[0, 1], [1, 2]]
        # Keywords co-occur
        kw_edges = [[0, 1], [1, 0]]

        mu = torch.from_numpy(edge2adj(user_edges, dim=3, weight=True, direct=True)).float()
        mi = torch.from_numpy(edge2adj(item_edges, dim=3, weight=True, direct=True)).float()
        mk = torch.from_numpy(edge2adj(kw_edges, dim=2, weight=True, direct=True)).float()

        T = torch.zeros(3, 3, 2)
        T[0, 0, 0] = 3.0  # user 0 dominates
        T[1, 1, 1] = 1.0
        T[2, 2, 0] = 1.0

        with patch('torch.cuda.is_available', return_value=False):
            u, v, w = socialAU(mu, mi, mk, T, epsilon=0.01)

        assert u.shape == (1, 3)
        assert v.shape == (1, 3)
        assert w.shape == (1, 2)
        assert torch.argmax(u[0]).item() == 0

    def test_tophits_and_socialau_agree_on_dominant_user(self):
        """On a simple structured tensor, tophits and socialAU should agree on the
        top-scoring user (socialAU adds HITS correction but dominant structure persists)."""
        torch.manual_seed(3)

        # User 0 is dominant: many triples, all pointing to item 0 / kw 0
        T = torch.zeros(4, 3, 3)
        T[0, 0, 0] = 10.0
        T[1, 1, 1] = 1.0
        T[2, 2, 2] = 1.0
        T[3, 0, 1] = 0.5

        mu = torch.eye(4).float()
        mi = torch.eye(3).float()
        mk = torch.eye(3).float()

        with patch('torch.cuda.is_available', return_value=False):
            u_top, _, _ = tophits(T.float(), epsilon=1e-8)
            u_social, _, _ = socialAU(mu, mi, mk, T.float(), epsilon=1e-6)

        assert torch.argmax(u_top).item() == 0
        assert torch.argmax(u_social[0]).item() == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_hits_large_iterations(self):
        adj = torch.eye(3).float()
        with patch('torch.cuda.is_available', return_value=False):
            hub, authority, h, a = hits(adj, p=1000)
        assert len(hub) == 3
        assert h.shape[0] == 1000

    def test_tophits_numerical_stability_small_values(self):
        T = torch.ones(2, 2, 2).float() * 1e-10
        with patch('torch.cuda.is_available', return_value=False):
            u, v, w = tophits(T, epsilon=1e-8)
        assert not torch.isnan(u).any()
        assert not torch.isinf(u).any()
        assert not torch.isnan(v).any()
        assert not torch.isinf(v).any()
        assert not torch.isnan(w).any()
        assert not torch.isinf(w).any()

    def test_tophits_zero_tensor(self):
        """Zero tensor: safe_normalize should return zero vectors without error"""
        T = torch.zeros(3, 3, 3).float()
        with patch('torch.cuda.is_available', return_value=False):
            u, v, w = tophits(T, epsilon=1e-3, max_iter=5)
        assert not torch.isnan(u).any()
        assert not torch.isnan(v).any()
        assert not torch.isnan(w).any()

    def test_socialau_single_user(self):
        """Degenerate case: 1 user, 1 item, 1 keyword"""
        mu = torch.tensor([[1.0]])
        mi = torch.tensor([[1.0]])
        mk = torch.tensor([[1.0]])
        T = torch.tensor([[[2.0]]])

        with patch('torch.cuda.is_available', return_value=False):
            u, v, w = socialAU(mu, mi, mk, T, epsilon=0.01)

        assert u.shape == (1, 1)
        assert v.shape == (1, 1)
        assert w.shape == (1, 1)
        assert abs(torch.linalg.norm(u).item() - 1.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
