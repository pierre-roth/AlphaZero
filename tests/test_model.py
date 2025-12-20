"""
Tests for model.py - AlphaZero Neural Network.
"""

import pytest
import torch
import numpy as np

from src.model import AlphaZeroNet, SEBlock, SEResidualBlock


class TestSEBlock:
    """Tests for Squeeze-and-Excitation block."""
    
    def test_output_shape(self):
        """SE block should preserve spatial dimensions."""
        se = SEBlock(channels=64, se_ratio=8)
        x = torch.randn(2, 64, 8, 8)
        out = se(x)
        
        assert out.shape == x.shape
    
    def test_different_batch_sizes(self):
        """Should work with different batch sizes."""
        se = SEBlock(channels=128, se_ratio=8)
        
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 128, 8, 8)
            out = se(x)
            assert out.shape == x.shape


class TestSEResidualBlock:
    """Tests for SE-Residual block."""
    
    def test_output_shape(self):
        """Residual block should preserve dimensions."""
        block = SEResidualBlock(channels=256, se_ratio=8)
        x = torch.randn(2, 256, 8, 8)
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_skip_connection(self):
        """Block should include skip connection."""
        block = SEResidualBlock(channels=64, se_ratio=8)
        
        # With all zeros, skip connection should dominate
        x = torch.zeros(1, 64, 8, 8)
        out = block(x)
        
        # Output should be non-negative due to final ReLU
        assert (out >= 0).all()


class TestAlphaZeroNet:
    """Tests for the full network."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return AlphaZeroNet(
            num_blocks=2,
            num_filters=32,
            num_input_planes=3,
            num_actions=192,
            se_ratio=8
        )
    
    def test_output_shapes(self, model):
        """Policy and WL heads should have correct shapes."""
        x = torch.randn(4, 3, 8, 8)
        policy, wl = model(x)
        
        assert policy.shape == (4, 192)
        assert wl.shape == (4, 2)
    
    def test_policy_logits(self, model):
        """Policy should output raw logits (not probabilities)."""
        x = torch.randn(2, 3, 8, 8)
        policy, _ = model(x)
        
        # Logits can be any real number
        assert policy.requires_grad
        
        # Softmax should give valid probabilities
        probs = torch.softmax(policy, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    
    def test_wl_logits(self, model):
        """WL should output 2 logits."""
        x = torch.randn(2, 3, 8, 8)
        _, wl = model(x)
        
        assert wl.shape[1] == 2
        
        # Softmax should give valid WL probabilities
        probs = torch.softmax(wl, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    
    def test_get_value(self, model):
        """get_value should convert WL to scalar."""
        x = torch.randn(2, 3, 8, 8)
        _, wl = model(x)
        
        value = model.get_value(wl)
        
        assert value.shape == (2, 1)
        assert (value >= -1).all()
        assert (value <= 1).all()
    
    def test_batch_size_one(self, model):
        """Should work with batch size 1."""
        x = torch.randn(1, 3, 8, 8)
        policy, wl = model(x)
        
        assert policy.shape == (1, 192)
        assert wl.shape == (1, 2)
    
    def test_eval_mode(self, model):
        """Model should work in eval mode."""
        model.eval()
        x = torch.randn(2, 3, 8, 8)
        
        with torch.no_grad():
            policy, wl = model(x)
        
        assert policy.shape == (2, 192)
        assert wl.shape == (2, 2)





class TestAlphaZeroNetGradients:
    """Tests for gradient flow."""
    
    def test_gradients_flow(self):
        """Gradients should flow through the network."""
        model = AlphaZeroNet(num_blocks=2, num_filters=32)
        x = torch.randn(2, 3, 8, 8)
        
        policy, wl = model(x)
        loss = policy.sum() + wl.sum()
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_policy_loss(self):
        """Cross-entropy loss should work for policy."""
        model = AlphaZeroNet(num_blocks=2, num_filters=32)
        x = torch.randn(2, 3, 8, 8)
        
        policy, _ = model(x)
        
        # Create random target distribution
        target = torch.softmax(torch.randn(2, 192), dim=1)
        
        # Cross-entropy style loss
        log_probs = torch.log_softmax(policy, dim=1)
        loss = -(target * log_probs).sum(dim=1).mean()
        
        loss.backward()
        assert model.policy_fc.weight.grad is not None
    
    def test_wl_loss(self):
        """Cross-entropy loss should work for WL."""
        model = AlphaZeroNet(num_blocks=2, num_filters=32)
        x = torch.randn(2, 3, 8, 8)
        
        _, wl = model(x)
        
        # Create WL targets (Win/Loss)
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        loss = torch.nn.functional.cross_entropy(wl, target)
        loss.backward()
        
        assert model.value_fc2.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
