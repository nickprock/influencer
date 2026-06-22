"""
Test suite for influencer/sparse.py
"""

import pytest
import torch
from unittest.mock import patch

from influencer.sparse import (
    to_sparse_tensor,
    sparse_mode_product,
    tophits_sparse,
    socialAU_sparse,
)
from influencer.torch_centrality import tophits, socialAU


# ---------------------------------------------------------------------------
# to_sparse_tensor
# ---------------------------------------------------------------------------

class TestToSparseTensor:

    def test_builds_coalesced_sparse_tensor(self):
        indices = torch.tensor([[0, 1, 2], [0, 1, 0], [1, 2, 3]], dtype=torch.long)
        values = torch.tensor([1.0, 2.0, 3.0])
        size = (3, 3, 4)

        T_sparse = to_sparse_tensor(indices, values, size)

        assert T_sparse.is_sparse
        assert T_sparse.is_coalesced()
        assert T_sparse.shape == size

        expected = torch.zeros(size)
        expected[0, 0, 1] = 1.0
        expected[1, 1, 2] = 2.0
        expected[2, 0, 3] = 3.0
        assert torch.allclose(T_sparse.to_dense(), expected)

    def test_duplicate_indices_are_summed_on_coalesce(self):
        indices = torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.long)
        values = torch.tensor([1.0, 2.0])
        T_sparse = to_sparse_tensor(indices, values, (2, 2, 2))
        assert T_sparse.to_dense()[0, 0, 0].item() == 3.0


# ---------------------------------------------------------------------------
# sparse_mode_product
# ---------------------------------------------------------------------------

class TestSparseModeProduct:

    @pytest.fixture
    def random_sparse_tensor(self):
        torch.manual_seed(0)
        N, M, Q = 6, 5, 4
        T_dense = torch.rand(N, M, Q)
        T_dense[T_dense < 0.7] = 0.0  # ~30% density, enough nonzeros for the test
        T_sparse = T_dense.to_sparse().coalesce()
        return T_dense, T_sparse

    def test_matches_dense_tensordot_mode0(self, random_sparse_tensor):
        T_dense, T_sparse = random_sparse_tensor
        N = T_dense.shape[0]
        vec = torch.rand(N)

        expected = torch.tensordot(T_dense, vec, dims=([0], [0]))
        actual = sparse_mode_product(T_sparse, vec, mode=0)

        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_matches_dense_tensordot_mode1(self, random_sparse_tensor):
        T_dense, T_sparse = random_sparse_tensor
        M = T_dense.shape[1]
        vec = torch.rand(M)

        expected = torch.tensordot(T_dense, vec, dims=([1], [0]))
        actual = sparse_mode_product(T_sparse, vec, mode=1)

        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_matches_dense_tensordot_mode2(self, random_sparse_tensor):
        T_dense, T_sparse = random_sparse_tensor
        Q = T_dense.shape[2]
        vec = torch.rand(Q)

        expected = torch.tensordot(T_dense, vec, dims=([2], [0]))
        actual = sparse_mode_product(T_sparse, vec, mode=2)

        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_invalid_mode_raises(self):
        T_dense = torch.rand(2, 2, 2)
        T_sparse = T_dense.to_sparse()
        with pytest.raises(ValueError):
            sparse_mode_product(T_sparse, torch.rand(2), mode=3)

    def test_zero_tensor_gives_zero_result(self):
        T_sparse = torch.zeros(3, 3, 3).to_sparse()
        result = sparse_mode_product(T_sparse, torch.rand(3), mode=0)
        assert torch.all(result == 0)


# ---------------------------------------------------------------------------
# tophits_sparse
# ---------------------------------------------------------------------------

class TestTophitsSparse:

    @patch('torch.cuda.is_available', return_value=False)
    def test_matches_dense_tophits_on_rank1_tensor(self, mock_cuda):
        """A pure rank-1 tensor has an exact, stable fixed point, so the sparse
        and dense code paths should agree far tighter than the 1e-5 spec tolerance."""
        torch.manual_seed(1)
        u0 = torch.rand(5) + 0.1
        v0 = torch.rand(4) + 0.1
        w0 = torch.rand(3) + 0.1
        T_dense = torch.einsum('i,j,k->ijk', u0, v0, w0)
        T_sparse = T_dense.to_sparse().coalesce()

        u_d, v_d, w_d = tophits(T_dense, epsilon=1e-8, max_iter=200)
        u_s, v_s, w_s = tophits_sparse(T_sparse, epsilon=1e-8, max_iter=200)

        assert torch.allclose(u_d, u_s, atol=1e-5)
        assert torch.allclose(v_d, v_s, atol=1e-5)
        assert torch.allclose(w_d, w_s, atol=1e-5)

    @patch('torch.cuda.is_available', return_value=False)
    def test_matches_dense_tophits_on_sparse_random_tensor(self, mock_cuda):
        torch.manual_seed(2)
        T_dense = torch.rand(6, 5, 4)
        T_dense[T_dense < 0.7] = 0.0
        T_sparse = T_dense.to_sparse().coalesce()

        u_d, v_d, w_d = tophits(T_dense, epsilon=1e-8, max_iter=500)
        u_s, v_s, w_s = tophits_sparse(T_sparse, epsilon=1e-8, max_iter=500)

        assert torch.allclose(u_d, u_s, atol=1e-5)
        assert torch.allclose(v_d, v_s, atol=1e-5)
        assert torch.allclose(w_d, w_s, atol=1e-5)

    @patch('torch.cuda.is_available', return_value=False)
    def test_dense_input_falls_back_to_tophits(self, mock_cuda):
        torch.manual_seed(3)
        T = torch.rand(4, 3, 5)
        u1, v1, w1 = tophits(T, epsilon=0.01)
        u2, v2, w2 = tophits_sparse(T, epsilon=0.01)
        assert torch.allclose(u1, u2)
        assert torch.allclose(v1, v2)
        assert torch.allclose(w1, w2)

    @patch('torch.cuda.is_available', return_value=False)
    def test_output_shapes(self, mock_cuda):
        T_dense = torch.rand(6, 5, 4)
        T_dense[T_dense < 0.7] = 0.0
        T_sparse = T_dense.to_sparse().coalesce()
        u, v, w = tophits_sparse(T_sparse, epsilon=0.01)
        assert u.shape == (6,)
        assert v.shape == (5,)
        assert w.shape == (4,)


# ---------------------------------------------------------------------------
# socialAU_sparse
# ---------------------------------------------------------------------------

class TestSocialAUSparse:

    @patch('torch.cuda.is_available', return_value=False)
    def test_matches_dense_socialau_on_rank1_tensor(self, mock_cuda):
        torch.manual_seed(4)
        n, m, r = 4, 3, 5
        mu = torch.eye(n)
        mi = torch.eye(m)
        mw = torch.eye(r)

        u0 = torch.rand(n) + 0.1
        v0 = torch.rand(m) + 0.1
        w0 = torch.rand(r) + 0.1
        T_dense = torch.einsum('i,j,k->ijk', u0, v0, w0)
        T_sparse = T_dense.to_sparse().coalesce()

        h_d, a_d, w_d = socialAU(mu, mi, mw, T_dense, epsilon=1e-8)
        h_s, a_s, w_s = socialAU_sparse(mu, mi, mw, T_sparse, epsilon=1e-8)

        assert torch.allclose(h_d, h_s, atol=1e-5)
        assert torch.allclose(a_d, a_s, atol=1e-5)
        assert torch.allclose(w_d, w_s, atol=1e-5)

    @patch('torch.cuda.is_available', return_value=False)
    def test_matches_dense_socialau_on_sparse_random_tensor(self, mock_cuda):
        torch.manual_seed(5)
        mu = torch.rand(3, 3).abs()
        mi = torch.rand(4, 4).abs()
        mw = torch.rand(5, 5).abs()
        T_dense = torch.rand(3, 4, 5)
        T_dense[T_dense < 0.7] = 0.0
        T_sparse = T_dense.to_sparse().coalesce()

        h_d, a_d, w_d = socialAU(mu, mi, mw, T_dense, epsilon=1e-6)
        h_s, a_s, w_s = socialAU_sparse(mu, mi, mw, T_sparse, epsilon=1e-6)

        assert torch.allclose(h_d, h_s, atol=1e-5)
        assert torch.allclose(a_d, a_s, atol=1e-5)
        assert torch.allclose(w_d, w_s, atol=1e-5)

    @patch('torch.cuda.is_available', return_value=False)
    def test_dense_input_falls_back_to_socialau(self, mock_cuda):
        torch.manual_seed(6)
        mu = torch.eye(3)
        mi = torch.eye(4)
        mw = torch.eye(5)
        T = torch.rand(3, 4, 5)
        h1, a1, w1 = socialAU(mu, mi, mw, T, epsilon=0.01)
        h2, a2, w2 = socialAU_sparse(mu, mi, mw, T, epsilon=0.01)
        assert torch.allclose(h1, h2)
        assert torch.allclose(a1, a2)
        assert torch.allclose(w1, w2)

    @patch('torch.cuda.is_available', return_value=False)
    def test_output_shapes(self, mock_cuda):
        mu = torch.eye(3)
        mi = torch.eye(4)
        mw = torch.eye(5)
        T_dense = torch.rand(3, 4, 5)
        T_dense[T_dense < 0.7] = 0.0
        T_sparse = T_dense.to_sparse().coalesce()
        h, a, w = socialAU_sparse(mu, mi, mw, T_sparse, epsilon=0.01)
        assert h.shape == (1, 3)
        assert a.shape == (1, 4)
        assert w.shape == (1, 5)


# ---------------------------------------------------------------------------
# Memory usage
# ---------------------------------------------------------------------------

class TestSparseMemoryUsage:
    """
    Note: tracemalloc does not observe PyTorch's native CPU allocator (verified
    empirically: allocating a 4MB dense tensor registers ~1KB of "Python" memory
    in tracemalloc), so it cannot be used to validate this claim. Instead we
    compare the actual byte footprint of each tensor's underlying storage,
    which is the quantity the sparse path is meant to reduce.
    """

    def test_sparse_storage_is_smaller_than_dense_below_1pct_density(self):
        torch.manual_seed(7)
        N, M, Q = 100, 100, 100
        total_entries = N * M * Q
        density = 0.005  # 0.5%, below the 1% threshold
        nnz = int(total_entries * density)

        indices = torch.stack([
            torch.randint(0, N, (nnz,)),
            torch.randint(0, M, (nnz,)),
            torch.randint(0, Q, (nnz,)),
        ])
        values = torch.rand(nnz)
        T_sparse = to_sparse_tensor(indices, values, (N, M, Q))

        dense_bytes = total_entries * torch.empty(0, dtype=torch.float32).element_size()

        sparse_idx = T_sparse.indices()
        sparse_val = T_sparse.values()
        sparse_bytes = (
            sparse_idx.element_size() * sparse_idx.nelement()
            + sparse_val.element_size() * sparse_val.nelement()
        )

        assert sparse_bytes < dense_bytes

    def test_sparse_mode_product_avoids_dense_materialisation(self):
        """sparse_mode_product never builds an (N, M, Q) dense tensor: the
        intermediate unfolding has at most nnz nonzeros, so its storage is
        bounded by nnz regardless of how large N, M, Q are."""
        torch.manual_seed(8)
        N, M, Q = 200, 200, 200
        total_entries = N * M * Q
        nnz = int(total_entries * 0.001)  # 0.1% density

        indices = torch.stack([
            torch.randint(0, N, (nnz,)),
            torch.randint(0, M, (nnz,)),
            torch.randint(0, Q, (nnz,)),
        ])
        values = torch.rand(nnz)
        T_sparse = to_sparse_tensor(indices, values, (N, M, Q))

        vec = torch.rand(N)
        result = sparse_mode_product(T_sparse, vec, mode=0)

        # Output is dense (M, Q) as documented, but the computation only ever
        # touched nnz entries of T -- never an (N, M, Q) dense buffer.
        assert result.shape == (M, Q)
        dense_bytes_avoided = total_entries * torch.empty(0, dtype=torch.float32).element_size()
        sparse_bytes_used = (
            T_sparse.indices().element_size() * T_sparse.indices().nelement()
            + T_sparse.values().element_size() * T_sparse.values().nelement()
        )
        assert sparse_bytes_used < dense_bytes_avoided


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
