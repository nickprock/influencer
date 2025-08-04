"""
Test suite for edge2adj.py and torch_centrality.py modules

Created for testing graph algorithms and centrality measures
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Import the modules to test
# Assuming the modules are in the same directory or properly installed
try:
    from influencer.edge2adj import edge2adj
    from influencer.torch_centrality import hits, tophits, socialAU
except ImportError:
    # If modules are not in path, you might need to adjust the import
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from influencer.edge2adj import edge2adj
    from influencer.torch_centrality import hits, tophits, socialAU


class TestEdge2Adj:
    """Test cases for the edge2adj function"""
    
    def test_basic_directed_weighted(self):
        """Test basic functionality with directed weighted graph"""
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
        """Test directed unweighted graph"""
        edgelist = [[0, 1], [1, 2], [0, 1]]  # Duplicate edge
        adj = edge2adj(edgelist=edgelist, dim=3, weight=False, direct=True)
        
        expected = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        assert np.array_equal(adj, expected)
    
    def test_basic_undirected_weighted(self):
        """Test undirected weighted graph"""
        edgelist = [[0, 1], [1, 2]]
        adj = edge2adj(edgelist=edgelist, dim=3, weight=True, direct=False)
        
        expected = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        assert np.array_equal(adj, expected)
    
    def test_basic_undirected_unweighted(self):
        """Test undirected unweighted graph"""
        edgelist = [[0, 1], [0, 1], [1, 2]]  # Duplicate edge
        adj = edge2adj(edgelist=edgelist, dim=3, weight=False, direct=False)
        
        expected = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        assert np.array_equal(adj, expected)
    
    def test_weighted_multiple_edges(self):
        """Test that multiple edges increase weight"""
        edgelist = [[0, 1], [0, 1], [0, 1]]
        adj = edge2adj(edgelist=edgelist, dim=2, weight=True, direct=True)
        
        expected = np.array([
            [0, 3],
            [0, 0]
        ])
        
        assert np.array_equal(adj, expected)
    
    def test_empty_edgelist(self):
        """Test with empty edge list"""
        adj = edge2adj(edgelist=[], dim=3, weight=True, direct=True)
        expected = np.zeros((3, 3), dtype=int)
        
        assert np.array_equal(adj, expected)
    
    def test_self_loops(self):
        """Test with self-loops"""
        edgelist = [[0, 0], [1, 1]]
        adj = edge2adj(edgelist=edgelist, dim=2, weight=True, direct=True)
        
        expected = np.array([
            [1, 0],
            [0, 1]
        ])
        
        assert np.array_equal(adj, expected)
    
    def test_large_dimension(self):
        """Test with larger matrix"""
        edgelist = [[0, 9], [5, 3], [9, 0]]
        adj = edge2adj(edgelist=edgelist, dim=10, weight=True, direct=True)
        
        assert adj.shape == (10, 10)
        assert adj[0, 9] == 1
        assert adj[5, 3] == 1
        assert adj[9, 0] == 1
        assert np.sum(adj) == 3
    
    def test_invalid_edge_indices(self):
        """Test behavior with edge indices that exceed dimension"""
        # This might raise an error or be silently ignored depending on implementation
        # The current implementation doesn't have bounds checking
        edgelist = [[0, 5]]  # Index 5 is out of bounds for dim=3
        with pytest.raises(IndexError):
            edge2adj(edgelist=edgelist, dim=3, weight=True, direct=True)


class TestHits:
    """Test cases for the HITS algorithm"""
    
    @pytest.fixture
    def sample_adj_matrix(self):
        """Create a sample adjacency matrix for testing"""
        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 0]
        ], dtype=float)
        return torch.from_numpy(adj).float()
    
    @patch('torch.cuda.is_available')
    def test_hits_basic_cpu(self, mock_cuda, sample_adj_matrix):
        """Test HITS algorithm on CPU"""
        mock_cuda.return_value = False
        
        hub, authority, h, a = hits(sample_adj_matrix, p=10)
        
        # Check that we get dictionaries with correct keys
        assert isinstance(hub, dict)
        assert isinstance(authority, dict)
        assert len(hub) == 4
        assert len(authority) == 4
        
        # Check that all scores are between 0 and 1 (normalized)
        for i in range(4):
            assert str(i) in hub
            assert str(i) in authority
            assert 0 <= hub[str(i)] <= 1
            assert 0 <= authority[str(i)] <= 1
        
        # Check tensor outputs
        assert h.shape[0] == 10
        assert a.shape[0] == 10
        assert h.shape[1] == 4  # number of nodes
        assert a.shape[1] == 4
    
    @patch('torch.cuda.is_available')
    def test_hits_convergence(self, mock_cuda, sample_adj_matrix):
        """Test that HITS scores are normalized"""
        mock_cuda.return_value = False
        
        hub, authority, h, a = hits(sample_adj_matrix, p=50)
        
        # Final scores should be normalized (sum of squares = 1)
        hub_values = [hub[str(i)] for i in range(4)]
        auth_values = [authority[str(i)] for i in range(4)]
        
        hub_norm = sum(v**2 for v in hub_values)
        auth_norm = sum(v**2 for v in auth_values)
        
        assert abs(hub_norm - 1.0) < 1e-5
        assert abs(auth_norm - 1.0) < 1e-5
    
    @patch('torch.cuda.is_available')
    def test_hits_single_node(self, mock_cuda):
        """Test HITS with single node"""
        mock_cuda.return_value = False
        
        adj = torch.zeros(1, 1).float()
        hub, authority, h, a = hits(adj, p=5)
        
        assert len(hub) == 1
        assert len(authority) == 1
        assert hub['0'] == authority['0']  # Should be same for single node
    
    @patch('torch.cuda.is_available') 
    def test_hits_gpu_fallback(self, mock_cuda):
        """Test that GPU fallback works correctly"""
        mock_cuda.return_value = False
        
        adj = torch.eye(3).float()
        hub, authority, h, a = hits(adj, p=5, device=0)
        
        # Should work without error even when GPU requested but not available
        assert isinstance(hub, dict)
        assert isinstance(authority, dict)


class TestTopHits:
    """Test cases for the TOPHITS algorithm"""
    
    @pytest.fixture
    def sample_3d_tensor(self):
        """Create a sample 3D tensor for testing"""
        # Create a small 3D binary tensor
        tensor = torch.zeros(3, 4, 5).float()
        tensor[0, 1, 2] = 1
        tensor[1, 2, 3] = 1
        tensor[2, 0, 1] = 1
        tensor[0, 3, 4] = 1
        return tensor
    
    @patch('torch.cuda.is_available')
    def test_tophits_basic(self, mock_cuda, sample_3d_tensor):
        """Test basic TOPHITS functionality"""
        mock_cuda.return_value = False
        
        u, v, w = tophits(sample_3d_tensor, epsilon=0.01)
        
        # Check output shapes
        assert u.shape == (sample_3d_tensor.shape[0],)
        assert v.shape == (sample_3d_tensor.shape[1],)
        assert w.shape == (sample_3d_tensor.shape[2],)

        
        # Check that vectors are normalized
        assert abs(torch.linalg.norm(u) - 1.0) < 1e-5
        assert abs(torch.linalg.norm(v) - 1.0) < 1e-5
        assert abs(torch.linalg.norm(w) - 1.0) < 1e-5
    
    @patch('torch.cuda.is_available')
    def test_tophits_convergence(self, mock_cuda):
        """Test TOPHITS convergence with different epsilon values"""
        mock_cuda.return_value = False
        
        tensor = torch.rand(4, 3, 5).float()
        tensor[tensor > 0.5] = 1
        tensor[tensor <= 0.5] = 0
        
        # Test with strict convergence
        u1, v1, w1 = tophits(tensor, epsilon=1e-6)
        
        # Test with loose convergence
        u2, v2, w2 = tophits(tensor, epsilon=0.1)
        
        # Both should produce normalized vectors
        assert abs(torch.linalg.norm(u1) - 1.0) < 1e-5
        assert abs(torch.linalg.norm(u2) - 1.0) < 1e-5


class TestSocialAU:
    """Test cases for the SocialAU algorithm"""
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for SocialAU"""
        # Create small adjacency matrices
        mu = torch.eye(3).float()  # Users network
        mi = torch.eye(4).float()  # Items network  
        mw = torch.eye(5).float()  # Keywords network
        
        # Create 3D tensor
        T = torch.rand(3, 4, 5).float()
        T[T > 0.7] = 1
        T[T <= 0.7] = 0
        
        return mu, mi, mw, T
    
    @patch('torch.cuda.is_available')
    def test_socialau_basic(self, mock_cuda, sample_inputs):
        """Test basic SocialAU functionality"""
        mock_cuda.return_value = False
        
        mu, mi, mw, T = sample_inputs
        u, v, w = socialAU(mu, mi, mw, T, epsilon=0.01)
        
        # Check output shapes
        assert u.shape == (1, 3)  # Users dimension
        assert v.shape == (1, 4)  # Items dimension
        assert w.shape == (1, 5)  # Keywords dimension
        
        # Check that vectors are normalized
        assert abs(torch.linalg.norm(u) - 1.0) < 1e-5
        assert abs(torch.linalg.norm(v) - 1.0) < 1e-5
        assert abs(torch.linalg.norm(w) - 1.0) < 1e-5
    
    @patch('torch.cuda.is_available')
    def test_socialau_convergence(self, mock_cuda, sample_inputs):
        """Test SocialAU convergence with different epsilon values"""
        mock_cuda.return_value = False
        
        mu, mi, mw, T = sample_inputs
        
        # Test with strict convergence
        u1, v1, w1 = socialAU(mu, mi, mw, T, epsilon=1e-6)
        
        # Test with loose convergence  
        u2, v2, w2 = socialAU(mu, mi, mw, T, epsilon=0.1)
        
        # Both should produce normalized vectors
        assert abs(torch.linalg.norm(u1) - 1.0) < 1e-5
        assert abs(torch.linalg.norm(u2) - 1.0) < 1e-5
    
    @patch('torch.cuda.is_available')
    def test_socialau_dimension_mismatch(self, mock_cuda):
        """Test SocialAU with mismatched dimensions"""
        mock_cuda.return_value = False
        
        mu = torch.eye(3).float()
        mi = torch.eye(3).float()  # Wrong size
        mw = torch.eye(3).float()  # Wrong size
        T = torch.rand(3, 4, 5).float()  # Doesn't match mi, mw
        
        # This should raise an error due to dimension mismatch
        with pytest.raises((RuntimeError, IndexError)):
            socialAU(mu, mi, mw, T, epsilon=0.01)


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_edge2adj_to_hits_pipeline(self):
        """Test pipeline from edgelist to HITS scores"""
        # Create edgelist
        edgelist = [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]]
        
        # Convert to adjacency matrix
        adj_np = edge2adj(edgelist, dim=3, weight=True, direct=True)
        
        # Convert to torch tensor
        adj_torch = torch.from_numpy(adj_np).float()
        
        # Run HITS
        with patch('torch.cuda.is_available', return_value=False):
            hub, authority, _, _ = hits(adj_torch, p=20)
        
        # Verify results
        assert len(hub) == 3
        assert len(authority) == 3
        
        # All nodes should have some hub/authority scores
        for i in range(3):
            assert str(i) in hub
            assert str(i) in authority
            assert hub[str(i)] >= 0
            assert authority[str(i)] >= 0
    
    def test_different_graph_types(self):
        """Test various graph configurations"""
        test_cases = [
            # (edgelist, dim, weight, direct, description)
            ([[0, 1], [1, 0]], 2, True, True, "bidirectional"),
            ([[0, 1], [1, 2], [2, 0]], 3, False, False, "undirected triangle"),
            ([[0, 0]], 1, True, True, "self-loop"),
            ([], 3, True, True, "empty graph"),
        ]
        
        for edgelist, dim, weight, direct, desc in test_cases:
            adj_np = edge2adj(edgelist, dim=dim, weight=weight, direct=direct)
            
            # Basic checks
            assert adj_np.shape == (dim, dim)
            assert adj_np.dtype == int
            
            # Convert to torch and test HITS (if graph has edges)
            if len(edgelist) > 0:
                adj_torch = torch.from_numpy(adj_np).float()
                with patch('torch.cuda.is_available', return_value=False):
                    hub, authority, _, _ = hits(adj_torch, p=10)
                    assert len(hub) == dim
                    assert len(authority) == dim


# Performance and edge case tests
class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_large_iterations(self):
        """Test algorithms with large iteration counts"""
        adj = torch.eye(3).float()
        
        with patch('torch.cuda.is_available', return_value=False):
            # Should not crash with large iteration count
            hub, authority, h, a = hits(adj, p=1000)
            assert len(hub) == 3
            assert h.shape[0] == 1000 
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Create tensor with very small values
        T = torch.ones(2, 2, 2).float() * 1e-10
        
        with patch('torch.cuda.is_available', return_value=False):
            u, v, w = tophits(T, epsilon=1e-8)
            
            # Should not produce NaN or Inf
            assert not torch.isnan(u).any()
            assert not torch.isinf(u).any()
            assert not torch.isnan(v).any()
            assert not torch.isinf(v).any()
            assert not torch.isnan(w).any()
            assert not torch.isinf(w).any()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])