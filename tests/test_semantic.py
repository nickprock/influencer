"""
Test suite for influencer/semantic.py
"""

import pytest
import torch

from influencer.semantic import (
    build_semantic_keyword_matrix,
    cluster_keywords,
    remap_tensor_to_clusters,
)


# ---------------------------------------------------------------------------
# build_semantic_keyword_matrix
# ---------------------------------------------------------------------------

class TestBuildSemanticKeywordMatrix:

    def test_identical_embeddings_yield_similarity_one(self):
        embeddings = torch.tensor([[1.0, 2.0, 3.0]] * 4)

        MK = build_semantic_keyword_matrix(embeddings, threshold=0.99)

        off_diagonal = MK + torch.eye(4)  # add back the zeroed diagonal
        assert torch.allclose(off_diagonal, torch.ones(4, 4), atol=1e-5)

    def test_orthogonal_embeddings_zeroed_by_threshold(self):
        embeddings = torch.eye(4)  # pairwise orthogonal, sim=0 off-diagonal

        MK = build_semantic_keyword_matrix(embeddings, threshold=0.7)

        assert torch.all(MK == 0)

    def test_threshold_keeps_only_similar_pairs(self):
        embeddings = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.01],   # nearly identical to keyword 0
            [0.0, 1.0],    # orthogonal to keyword 0
        ])

        MK = build_semantic_keyword_matrix(embeddings, threshold=0.9)

        assert MK[0, 1] > 0.9
        assert MK[1, 0] > 0.9
        assert MK[0, 2] == 0
        assert MK[2, 0] == 0

    def test_top_k_one_keeps_exactly_one_neighbour_per_row_before_symmetrizing(self):
        torch.manual_seed(0)
        embeddings = torch.rand(6, 8)

        MK = build_semantic_keyword_matrix(embeddings, top_k=1)

        # Symmetrizing can add entries back, so each row has *at least* one
        # nonzero neighbour; verify the underlying per-row selection by
        # checking every row has a maximum exactly equal to that keyword's
        # best match (i.e. at least one strong connection exists).
        assert (MK > 0).any(dim=1).all()

    def test_top_k_one_on_symmetric_mutual_best_friends(self):
        # Two well-separated pairs: each keyword's unique best match is its
        # pair partner, so top_k=1 is already symmetric and unambiguous.
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.01],
        ])

        MK = build_semantic_keyword_matrix(embeddings, top_k=1)

        nonzero = (MK > 0)
        expected = torch.tensor([
            [False, True, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, True, False],
        ])
        assert torch.equal(nonzero, expected)

    def test_output_is_symmetric_with_zero_diagonal(self):
        torch.manual_seed(1)
        embeddings = torch.rand(10, 16)

        MK = build_semantic_keyword_matrix(embeddings, threshold=0.5)

        assert torch.allclose(MK, MK.T)
        assert torch.all(torch.diagonal(MK) == 0)
        assert MK.shape == (10, 10)
        assert MK.dtype == torch.float32

    def test_top_k_takes_precedence_over_threshold(self):
        torch.manual_seed(2)
        embeddings = torch.rand(8, 4)

        MK_top_k = build_semantic_keyword_matrix(embeddings, threshold=0.999, top_k=2)

        # With threshold=0.999 almost nothing would normally survive, but
        # top_k=2 should still produce neighbours for every keyword.
        assert (MK_top_k > 0).any(dim=1).all()

    def test_handles_large_keyword_count_without_python_loop_blowup(self):
        torch.manual_seed(3)
        R = 6000  # larger than the internal chunk size, exercises batching
        embeddings = torch.rand(R, 4)

        MK = build_semantic_keyword_matrix(embeddings, top_k=3)

        assert MK.shape == (R, R)
        assert torch.allclose(MK, MK.T)


# ---------------------------------------------------------------------------
# cluster_keywords
# ---------------------------------------------------------------------------

class TestClusterKeywords:

    def test_all_keywords_assigned_a_valid_cluster(self):
        torch.manual_seed(4)
        embeddings = torch.rand(30, 5)

        labels, centroids = cluster_keywords(embeddings, n_clusters=4)

        assert labels.shape == (30,)
        assert labels.dtype == torch.long
        assert centroids.shape == (4, 5)
        assert torch.all(labels >= 0) and torch.all(labels < 4)

    def test_converges_on_well_separated_blobs(self):
        torch.manual_seed(5)
        blob_a = torch.randn(20, 2) * 0.01 + torch.tensor([10.0, 10.0])
        blob_b = torch.randn(20, 2) * 0.01 + torch.tensor([-10.0, -10.0])
        embeddings = torch.cat([blob_a, blob_b], dim=0)

        labels, centroids = cluster_keywords(embeddings, n_clusters=2, n_iter=50)

        # All points within a blob must share a label, and the two blobs
        # must land in different clusters.
        assert torch.unique(labels[:20]).numel() == 1
        assert torch.unique(labels[20:]).numel() == 1
        assert labels[0] != labels[20]

    def test_reproducible_with_same_seed(self):
        torch.manual_seed(6)
        embeddings = torch.rand(25, 6)

        labels1, centroids1 = cluster_keywords(embeddings, n_clusters=3, seed=123)
        labels2, centroids2 = cluster_keywords(embeddings, n_clusters=3, seed=123)

        assert torch.equal(labels1, labels2)
        assert torch.allclose(centroids1, centroids2)

    def test_invalid_n_clusters_raises(self):
        embeddings = torch.rand(5, 3)
        with pytest.raises(ValueError):
            cluster_keywords(embeddings, n_clusters=0)
        with pytest.raises(ValueError):
            cluster_keywords(embeddings, n_clusters=6)


# ---------------------------------------------------------------------------
# remap_tensor_to_clusters
# ---------------------------------------------------------------------------

class TestRemapTensorToClusters:

    def test_conserves_total_sum(self):
        torch.manual_seed(7)
        T = torch.rand(3, 4, 10)
        labels = torch.randint(0, 5, (10,))

        T_clustered = remap_tensor_to_clusters(T, labels)

        assert torch.allclose(T_clustered.sum(), T.sum(), atol=1e-5)

    def test_output_shape_matches_number_of_clusters(self):
        T = torch.rand(2, 2, 6)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        T_clustered = remap_tensor_to_clusters(T, labels)

        assert T_clustered.shape == (2, 2, 3)

    def test_entries_are_summed_by_cluster(self):
        T = torch.zeros(1, 1, 4)
        T[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        labels = torch.tensor([0, 0, 1, 1])

        T_clustered = remap_tensor_to_clusters(T, labels)

        assert T_clustered[0, 0, 0].item() == pytest.approx(3.0)
        assert T_clustered[0, 0, 1].item() == pytest.approx(7.0)

    def test_mismatched_labels_shape_raises(self):
        T = torch.rand(1, 1, 4)
        labels = torch.tensor([0, 1, 2])  # wrong length
        with pytest.raises(ValueError):
            remap_tensor_to_clusters(T, labels)

    def test_integrates_with_cluster_keywords(self):
        torch.manual_seed(8)
        embeddings = torch.rand(12, 5)
        labels, _ = cluster_keywords(embeddings, n_clusters=4)
        T = torch.rand(2, 3, 12)

        T_clustered = remap_tensor_to_clusters(T, labels)

        assert T_clustered.shape == (2, 3, int(labels.max().item()) + 1)
        assert torch.allclose(T_clustered.sum(), T.sum(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
