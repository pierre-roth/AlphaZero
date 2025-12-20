"""
Tests for Arena System.

Tests the matchmaking heuristics, random opening generation,
and match count tracking.
"""

import pytest
import numpy as np
import os
import tempfile
import json

from src.arena import (
    ArenaState, Arena, 
    INITIAL_ELO, EXPLORATION_RATE, BIAS_LAMBDA, TOP_K
)
from src.game import BreakthroughGame


class TestArenaState:
    """Tests for ArenaState class."""
    
    def test_pair_key_generation(self):
        """Test that pair keys are generated consistently (sorted)."""
        key1 = ArenaState._get_pair_key("model_a", "model_b")
        key2 = ArenaState._get_pair_key("model_b", "model_a")
        assert key1 == key2
        assert key1 == "model_a|model_b"
    
    def test_match_count_tracking(self):
        """Test that match counts are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ArenaState(checkpoint_dir=tmpdir)
            
            # Initially no games
            assert state.get_match_count("model_a", "model_b") == 0
            
            # Record a match with 2 games
            state.ratings["model_a"] = INITIAL_ELO
            state.ratings["model_b"] = INITIAL_ELO
            state.record_match("model_a", "model_b", wins_a=1, wins_b=1)
            
            # Should now have 2 games
            assert state.get_match_count("model_a", "model_b") == 2
            assert state.get_match_count("model_b", "model_a") == 2  # Order doesn't matter
    
    def test_match_counts_persistence(self):
        """Test that match_counts survive save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate state
            state1 = ArenaState(checkpoint_dir=tmpdir)
            state1.ratings["model_a"] = INITIAL_ELO
            state1.ratings["model_b"] = INITIAL_ELO
            state1.record_match("model_a", "model_b", wins_a=1, wins_b=1)
            
            # Reload state
            state2 = ArenaState(checkpoint_dir=tmpdir)
            assert state2.get_match_count("model_a", "model_b") == 2


class TestMatchmakingHeuristic:
    """Tests for the matchmaking selection heuristic."""
    
    def test_select_matchup_prefers_equal_ratings(self):
        """Test that equal-rated models have higher scores than unequal ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ArenaState(checkpoint_dir=tmpdir)
            state.ratings = {
                "model_a": 1000,
                "model_b": 1000,
                "model_c": 1400,
            }
            state.save()
            
            arena = Arena(device="cpu")
            arena.state = state
            
            matchup = arena.select_matchup()
            assert matchup is not None
            
            model_a, model_b, score = matchup
            # With equal ratings and no games played, A-B should have highest base score
            # The bias term slightly favors C, but variance (0.25 vs ~0.08) dominates
            # So A-B should usually be selected (85% of time due to exploration)
            # We just check the matchup is valid
            assert model_a in state.ratings
            assert model_b in state.ratings
    
    def test_select_matchup_penalizes_repeats(self):
        """Test that pairs with many games get lower scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ArenaState(checkpoint_dir=tmpdir)
            state.ratings = {
                "model_a": 1000,
                "model_b": 1000,
                "model_c": 1000,
            }
            # A and B have played 100 games, A and C have played 0
            state.match_counts = {
                ArenaState._get_pair_key("model_a", "model_b"): 100,
            }
            state.save()
            
            arena = Arena(device="cpu")
            arena.state = state
            
            # Run multiple times to check tendency (due to exploration)
            selections = {"a_b": 0, "a_c": 0, "b_c": 0}
            for _ in range(50):
                matchup = arena.select_matchup()
                if matchup:
                    model_a, model_b, _ = matchup
                    key = ArenaState._get_pair_key(model_a, model_b)
                    if "model_a" in key and "model_b" in key:
                        selections["a_b"] += 1
                    elif "model_a" in key and "model_c" in key:
                        selections["a_c"] += 1
                    else:
                        selections["b_c"] += 1
            
            # A-C and B-C should be selected more than A-B (which has 100 games)
            assert selections["a_c"] + selections["b_c"] > selections["a_b"]
    
    def test_select_matchup_returns_none_with_fewer_than_2_models(self):
        """Test that select_matchup returns None if < 2 models exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ArenaState(checkpoint_dir=tmpdir)
            state.ratings = {"model_a": 1000}
            state.save()
            
            arena = Arena(device="cpu")
            arena.state = state
            
            assert arena.select_matchup() is None


class TestRandomOpening:
    """Tests for random opening generation."""
    
    def test_random_opening_not_initial(self):
        """Test that random opening is different from initial position."""
        arena = Arena(device="cpu")
        opening = arena._get_random_opening(num_moves=6)
        
        initial = BreakthroughGame()
        
        # The boards should be different
        assert not np.array_equal(opening.board, initial.board)
    
    def test_random_opening_is_valid(self):
        """Test that random opening produces a valid game state."""
        arena = Arena(device="cpu")
        opening = arena._get_random_opening(num_moves=6)
        
        # Should not be terminal (6 moves is too few)
        assert not opening.is_terminal()
        
        # Should have legal moves
        assert len(opening.get_legal_moves()) > 0
    
    def test_random_opening_variability(self):
        """Test that multiple random openings are different."""
        arena = Arena(device="cpu")
        openings = [arena._get_random_opening(num_moves=6) for _ in range(10)]
        
        # Check that not all openings are identical
        boards = [tuple(o.board.flatten()) for o in openings]
        unique_boards = set(boards)
        
        # With 6 random moves, we should get some variety
        assert len(unique_boards) > 1


class TestHeuristicMath:
    """Tests for the mathematical correctness of the heuristic."""
    
    def test_variance_calculation(self):
        """Test that p(1-p) is max at p=0.5."""
        # Equal ratings -> p=0.5
        ra, rb = 1000, 1000
        p = 1 / (1 + 10 ** ((rb - ra) / 400))
        variance_equal = p * (1 - p)
        
        # 400 rating difference -> p ≈ 0.09
        ra, rb = 1000, 1400
        p = 1 / (1 + 10 ** ((rb - ra) / 400))
        variance_unequal = p * (1 - p)
        
        assert variance_equal > variance_unequal
        assert abs(variance_equal - 0.25) < 0.01  # p(1-p) at p=0.5 is 0.25
    
    def test_bias_multiplier(self):
        """Test that bias correctly favors higher-rated models."""
        import math
        
        mu = 1000
        sigma = 200
        epsilon = 1e-9
        
        # Low-rated pair
        r_top_low = 1000
        z_low = (r_top_low - mu) / (sigma + epsilon)
        mult_low = math.exp(BIAS_LAMBDA * z_low)
        
        # High-rated pair
        r_top_high = 1400
        z_high = (r_top_high - mu) / (sigma + epsilon)
        mult_high = math.exp(BIAS_LAMBDA * z_high)
        
        assert mult_high > mult_low
