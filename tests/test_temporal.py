"""
Test suite for influencer/temporal.py
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
import torch

from influencer.temporal import build_temporal_tensor, socialAU_temporal
from influencer.torch_centrality import socialAU


# ---------------------------------------------------------------------------
# build_temporal_tensor
# ---------------------------------------------------------------------------

class TestBuildTemporalTensor:

    def test_uniform_decay_reproduces_static_tensor(self):
        ref = datetime(2024, 1, 10)
        triples = [
            (0, 1, 2, ref),
            (0, 1, 2, ref - timedelta(days=3)),
            (1, 0, 1, ref - timedelta(days=5)),
        ]

        T = build_temporal_tensor(triples, n=2, m=2, r=3, window=7, decay=1.0, reference_time=ref)

        expected = torch.zeros(2, 2, 3)
        expected[0, 1, 2] = 2.0  # two events, decay=1.0 contributes 1.0 each
        expected[1, 0, 1] = 1.0
        assert torch.allclose(T, expected)

    def test_older_events_have_less_weight(self):
        ref = datetime(2024, 1, 10)
        triples = [
            (0, 0, 0, ref),                          # bucket 0 -> decay**0
            (1, 1, 1, ref - timedelta(days=2)),       # bucket 2 -> decay**2
        ]

        T = build_temporal_tensor(triples, n=2, m=2, r=2, window=7, decay=0.9, reference_time=ref)

        assert T[0, 0, 0].item() == pytest.approx(0.9 ** 0)
        assert T[1, 1, 1].item() == pytest.approx(0.9 ** 2)
        assert T[0, 0, 0].item() > T[1, 1, 1].item()

    def test_triples_outside_window_are_ignored(self):
        ref = datetime(2024, 1, 10)
        triples = [
            (0, 0, 0, ref - timedelta(days=10)),  # older than window=7
        ]

        T = build_temporal_tensor(triples, n=1, m=1, r=1, window=7, decay=0.9, reference_time=ref)

        assert torch.all(T == 0)

    def test_reference_time_defaults_to_max_timestamp(self):
        t0 = datetime(2024, 1, 10)
        t1 = datetime(2024, 1, 5)
        triples = [(0, 0, 0, t0), (0, 0, 0, t1)]

        T_default = build_temporal_tensor(triples, n=1, m=1, r=1, window=7, decay=0.9)
        T_explicit = build_temporal_tensor(triples, n=1, m=1, r=1, window=7, decay=0.9, reference_time=t0)

        assert torch.allclose(T_default, T_explicit)

    def test_numeric_timestamps_treated_as_time_unit_from_epoch(self):
        # reference defaults to max(100.0, 97.0) = 100.0; second event is 3 days old
        triples = [
            (0, 0, 0, 100.0),
            (0, 0, 0, 97.0),
        ]

        T = build_temporal_tensor(triples, n=1, m=1, r=1, window=7, decay=0.9, time_unit="days")

        expected = 0.9 ** 0 + 0.9 ** 3
        assert T[0, 0, 0].item() == pytest.approx(expected)

    def test_future_events_clamped_to_bucket_zero(self):
        ref = datetime(2024, 1, 10)
        triples = [(0, 0, 0, ref + timedelta(days=2))]

        T = build_temporal_tensor(triples, n=1, m=1, r=1, window=7, decay=0.9, reference_time=ref)

        assert T[0, 0, 0].item() == pytest.approx(0.9 ** 0)

    def test_invalid_time_unit_raises(self):
        ref = datetime(2024, 1, 10)
        with pytest.raises(ValueError):
            build_temporal_tensor([(0, 0, 0, ref)], n=1, m=1, r=1, time_unit="fortnights")

    def test_output_dtype_and_shape(self):
        ref = datetime(2024, 1, 10)
        T = build_temporal_tensor([(0, 0, 0, ref)], n=3, m=4, r=5, reference_time=ref)
        assert T.shape == (3, 4, 5)
        assert T.dtype == torch.float32


# ---------------------------------------------------------------------------
# socialAU_temporal
# ---------------------------------------------------------------------------

class TestSocialAUTemporal:

    @patch('torch.cuda.is_available', return_value=False)
    def test_matches_socialau_on_equivalent_static_tensor(self, mock_cuda):
        ref = datetime(2024, 1, 10)
        n, m, r = 3, 3, 3
        mu = torch.eye(n)
        mi = torch.eye(m)
        mw = torch.eye(r)
        triples = [(0, 1, 2, ref), (1, 2, 0, ref - timedelta(days=1))]

        u_t, v_t, w_t = socialAU_temporal(
            mu, mi, mw, triples, n, m, r,
            window=7, decay=1.0, reference_time=ref, epsilon=1e-6,
        )

        T_static = build_temporal_tensor(triples, n, m, r, window=7, decay=1.0, reference_time=ref)
        u_s, v_s, w_s = socialAU(mu, mi, mw, T_static, epsilon=1e-6)

        assert torch.allclose(u_t, u_s)
        assert torch.allclose(v_t, v_s)
        assert torch.allclose(w_t, w_s)

    @patch('torch.cuda.is_available', return_value=False)
    def test_output_shapes(self, mock_cuda):
        ref = datetime(2024, 1, 10)
        n, m, r = 4, 3, 5
        mu = torch.eye(n)
        mi = torch.eye(m)
        mw = torch.eye(r)
        triples = [(0, 0, 0, ref)]

        u, v, w = socialAU_temporal(mu, mi, mw, triples, n, m, r, reference_time=ref, epsilon=0.01)

        assert u.shape == (1, n)
        assert v.shape == (1, m)
        assert w.shape == (1, r)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
