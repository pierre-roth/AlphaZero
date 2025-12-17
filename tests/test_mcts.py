"""
Tests for mcts.py - Monte Carlo Tree Search.
"""

import pytest
import torch
import numpy as np

from src.mcts import MCTS, Node
from src.model import AlphaZeroNet
from src.game import BreakthroughGame, NUM_ACTIONS


class TestNode:
    """Tests for MCTS Node."""
    
    def test_initial_state(self):
        """New node should have zero visits and empty children."""
        node = Node(prior=0.5)
        
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.5
        assert len(node.children) == 0
    
    def test_value_zero_visits(self):
        """Value should be 0 with no visits."""
        node = Node(prior=0.5)
        assert node.value() == 0.0
    
    def test_value_with_visits(self):
        """Value should be average of accumulated values."""
        node = Node(prior=0.5)
        node.visit_count = 10
        node.value_sum = 5.0
        
        assert node.value() == 0.5
    
    def test_is_expanded(self):
        """is_expanded should check for children."""
        node = Node(prior=0.5)
        assert not node.is_expanded()
        
        node.children[0] = Node(prior=0.3)
        assert node.is_expanded()


class TestMCTS:
    """Tests for MCTS search."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        model = AlphaZeroNet(num_blocks=2, num_filters=32)
        model.eval()
        return model
    
    @pytest.fixture
    def mcts(self, model):
        """Create MCTS with test settings."""
        return MCTS(
            model=model,
            num_simulations=10,
            c_puct=1.5,
            fpu_value=0.0,
            device="cpu"
        )
    
    def test_search_returns_root(self, mcts):
        """Search should return a root node."""
        game = BreakthroughGame()
        root = mcts.search(game)
        
        assert isinstance(root, Node)
        assert root.is_expanded()
    
    def test_root_has_children(self, mcts):
        """Root should have children after search."""
        game = BreakthroughGame()
        root = mcts.search(game)
        
        # Should have some legal moves as children
        assert len(root.children) > 0
    
    def test_visit_counts_sum(self, mcts):
        """Total visits should be at least num_simulations."""
        game = BreakthroughGame()
        root = mcts.search(game)
        
        assert root.visit_count >= mcts.num_simulations
    
    def test_dirichlet_noise(self, mcts):
        """Adding noise should change priors."""
        game = BreakthroughGame()
        
        # Search without noise
        root1 = mcts.search(game, add_noise=False)
        priors1 = [child.prior for child in root1.children.values()]
        
        # Search with noise (new root)
        root2 = mcts.search(game, add_noise=True)
        priors2 = [child.prior for child in root2.children.values()]
        
        # Priors should be different (with high probability)
        assert priors1 != priors2 or len(priors1) == 1
    
    def test_action_probs_shape(self, mcts):
        """Action probs should have correct shape."""
        game = BreakthroughGame()
        root = mcts.search(game)
        
        probs = mcts.get_action_probs(root, temperature=1.0)
        
        assert probs.shape == (NUM_ACTIONS,)
        assert probs.dtype == np.float32
    
    def test_action_probs_sum_to_one(self, mcts):
        """Action probs should sum to 1."""
        game = BreakthroughGame()
        root = mcts.search(game)
        
        probs = mcts.get_action_probs(root, temperature=1.0)
        
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    
    def test_temperature_zero(self, mcts):
        """Temperature 0 should give deterministic (one-hot) probs."""
        game = BreakthroughGame()
        root = mcts.search(game)
        
        probs = mcts.get_action_probs(root, temperature=0.0)
        
        # Should be one-hot
        assert probs.max() == 1.0
        assert np.count_nonzero(probs) == 1
    
    def test_tree_reuse(self, mcts):
        """Providing an existing root should reuse the tree."""
        game = BreakthroughGame()
        
        # First search
        root = mcts.search(game)
        initial_visits = root.visit_count
        
        # Search again with same root
        root = mcts.search(game, root=root)
        
        # Should have more visits
        assert root.visit_count > initial_visits


class TestMCTSIntegration:
    """Integration tests for MCTS with real game play."""
    
    @pytest.fixture
    def mcts(self):
        """Create MCTS for testing."""
        model = AlphaZeroNet(num_blocks=2, num_filters=32)
        model.eval()
        return MCTS(model, num_simulations=20, device="cpu")
    
    def test_play_full_game_segment(self, mcts):
        """Should be able to play several moves without crashing."""
        game = BreakthroughGame()
        
        for _ in range(10):  # Play 10 moves
            if game.is_terminal():
                break
            
            root = mcts.search(game, add_noise=True)
            probs = mcts.get_action_probs(root, temperature=1.0)
            
            # Sample action
            action = int(np.random.choice(len(probs), p=probs))
            move = game.decode_action(action)
            game.step(move)
        
        # Should complete without error
        assert True
    
    def test_different_positions(self, mcts):
        """MCTS should work from various positions."""
        # Starting position
        game1 = BreakthroughGame()
        root1 = mcts.search(game1)
        assert root1.is_expanded()
        
        # After a few moves
        game2 = BreakthroughGame()
        for _ in range(4):
            moves = game2.get_legal_moves()
            game2.step(moves[0])
        
        root2 = mcts.search(game2)
        assert root2.is_expanded()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
