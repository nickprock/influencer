"""
Test suite for influencer/init_utils.py and socialAU's h_init/a_init/w_init
warm-start parameters.
"""

import pytest
import torch

from influencer.torch_centrality import socialAU
from influencer.init_utils import make_embedding_init


# ---------------------------------------------------------------------------
# socialAU warm-start (h_init / a_init / w_init)
# ---------------------------------------------------------------------------

class TestSocialAUWarmStart:

    @pytest.fixture
    def sample_inputs(self):
        torch.manual_seed(0)
        mu = torch.rand(3, 3).abs()
        mi = torch.rand(4, 4).abs()
        mw = torch.rand(5, 5).abs()
        T = torch.rand(3, 4, 5)
        return mu, mi, mw, T

    def test_h_init_ones_matches_default(self, sample_inputs):
        # a_init/w_init are deliberately left at their default here: unlike
        # h, the a/w recurrence reads their own previous value (a_prev,
        # old w), so normalising a provided a_init/w_init does shift the
        # result relative to the raw torch.ones default.
        mu, mi, mw, T = sample_inputs

        u_default, a_default, w_default = socialAU(mu, mi, mw, T, epsilon=1e-6, device="cpu")
        u_ones, a_ones, w_ones = socialAU(
            mu, mi, mw, T, epsilon=1e-6, device="cpu", h_init=torch.ones(3)
        )

        assert torch.allclose(u_default, u_ones)
        assert torch.allclose(a_default, a_ones)
        assert torch.allclose(w_default, w_ones)

    def test_wrong_shape_h_init_raises(self, sample_inputs):
        mu, mi, mw, T = sample_inputs
        with pytest.raises(ValueError):
            socialAU(mu, mi, mw, T, epsilon=1e-6, device="cpu", h_init=torch.ones(4))

    def test_wrong_shape_a_init_raises(self, sample_inputs):
        mu, mi, mw, T = sample_inputs
        with pytest.raises(ValueError):
            socialAU(mu, mi, mw, T, epsilon=1e-6, device="cpu", a_init=torch.ones(3))

    def test_wrong_shape_w_init_raises(self, sample_inputs):
        mu, mi, mw, T = sample_inputs
        with pytest.raises(ValueError):
            socialAU(mu, mi, mw, T, epsilon=1e-6, device="cpu", w_init=torch.ones(2))

    def test_content_driven_init_identifies_true_dominant_node(self):
        """
        socialAU's stop rule (lambda decrease <= epsilon, with lambda_prev
        starting at 0) makes it halt after exactly 2 iterations for
        virtually any input, so "fewer iterations" isn't a usable signal
        here. We instead check quality at equal effort (same epsilon, same
        2-iteration budget): node 0 is the true dominant user (one
        concentrated, on-topic post), but a rival has more scattered raw
        activity that outweighs it under an uninformed (flat) start. A
        content-based prior concentrated on node 0 corrects this within
        that same budget.
        """
        n = m = r = 6
        mu = torch.zeros(n, n)
        mi = torch.eye(m)
        mw = torch.eye(r)

        T = torch.zeros(n, m, r)
        T[0, 0, 0] = 1.0  # node 0: one concentrated, on-topic post
        for k in range(1, 6):  # rival: 5 scattered posts, larger raw total mass
            T[1, k, k] = 1.0

        u_flat, _, _ = socialAU(mu, mi, mw, T, epsilon=1e-6, device="cpu")
        assert torch.argmax(u_flat[0]).item() == 1, "flat start should wrongly favour the rival"

        peaked = torch.full((n,), 0.01)
        peaked[0] = 5.0
        u_warm, _, _ = socialAU(
            mu, mi, mw, T, epsilon=1e-6, device="cpu",
            h_init=peaked.clone(), a_init=peaked.clone(), w_init=peaked.clone(),
        )
        assert torch.argmax(u_warm[0]).item() == 0, "warm start should identify the true dominant node"


# ---------------------------------------------------------------------------
# make_embedding_init
# ---------------------------------------------------------------------------

class TestMakeEmbeddingInit:

    def test_norm_aggregation_returns_unit_vectors(self):
        embeddings = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])

        result = make_embedding_init(["a", "b", "c"], lambda texts: embeddings, aggregation="norm")

        assert result.shape == (3, 2)
        assert torch.allclose(result.norm(dim=1), torch.ones(3), atol=1e-5)

    def test_mean_aggregation_returns_one_scalar_per_text(self):
        torch.manual_seed(0)
        embeddings = torch.rand(5, 8)

        result = make_embedding_init(["a", "b", "c", "d", "e"], lambda texts: embeddings)

        assert result.shape == (5,)
        assert result.dtype == torch.float32

    def test_mean_aggregation_separates_distinct_clusters(self):
        # Two well-separated blobs: the first principal component should
        # cleanly rank one cluster above the other.
        blob_a = torch.tensor([[10.0, 0.0]] * 3)
        blob_b = torch.tensor([[-10.0, 0.0]] * 3)
        embeddings = torch.cat([blob_a, blob_b], dim=0)

        result = make_embedding_init(["t"] * 6, lambda texts: embeddings)

        assert torch.sign(result[0]) != torch.sign(result[3])
        assert torch.allclose(result[:3], result[0].expand(3), atol=1e-4)
        assert torch.allclose(result[3:], result[3].expand(3), atol=1e-4)

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError):
            make_embedding_init(["a", "b"], lambda texts: torch.rand(2, 4), aggregation="bogus")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
