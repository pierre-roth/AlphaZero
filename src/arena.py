"""
Arena System for Model Evaluation.

This module implements ELO-based rating for comparing model checkpoints.
It runs independently of training and evaluates models as they are generated.

Key features:
- Standard ELO rating updates (K=32)
- Persistent state in arena_state.json
- Automatic detection of new checkpoints
- Best model tracking (model_best.pt)
"""

import os
import json
import glob
import time
import random
import math
from itertools import combinations
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

from src.game import BreakthroughGame, WHITE, BLACK
from src.model import AlphaZeroNet
from src.mcts import MCTS
from src.config import Config


# ELO Constants
INITIAL_ELO = 1000
K_FACTOR = 32
GAMES_PER_MATCH = 20  # Games to play per model comparison

# Matchmaking Constants
EXPLORATION_RATE = 0.15  # Probability to pick from top-K instead of top-1
TOP_K = 5  # Size of candidate pool for exploration
BIAS_LAMBDA = 0.15  # Strength of higher-ranked bias
RANDOM_OPENING_MOVES = 6  # Number of random moves for paired openings


class ArenaState:
    """Persistent state for the arena."""
    
    def __init__(self, checkpoint_dir: str = Config.CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, "arena_state.json")
        
        self.ratings: Dict[str, float] = {}
        self.matches: List[dict] = []
        self.matches: List[dict] = []
        # Track best model (standard size)
        self.best_model: Optional[str] = None
        # Track total games between pairs for matchmaking heuristics
        # Key: "model_a|model_b" (sorted), Value: int (game count)
        self.match_counts: Dict[str, int] = {}
        
        self.load()
    
    def load(self):
        """Load state from disk."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.ratings = data.get('ratings', {})
                self.matches = data.get('matches', [])
                self.best_model = data.get('best_model')
                
                # Rebuild match_counts from historical data for consistency
                self._rebuild_match_counts()
            print(f"Loaded arena state: {len(self.ratings)} models rated, best: {self.best_model}")
        else:
            print("No existing arena state found, starting fresh")
    
    def _rebuild_match_counts(self):
        """Rebuild match_counts from matches and cross_size_matches for consistency."""
        self.match_counts = {}
        # Count from regular matches
        for match in self.matches:
            pair_key = self._get_pair_key(match['model_a'], match['model_b'])
            total_games = match['wins_a'] + match['wins_b']
            self.match_counts[pair_key] = self.match_counts.get(pair_key, 0) + total_games

    
    def save(self):
        """Save state to disk."""
        data = {
            'ratings': self.ratings,
            'matches': self.matches,
            'best_model': self.best_model,
            'match_counts': self.match_counts,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def _get_pair_key(model_a: str, model_b: str) -> str:
        """Get canonical key for a model pair (sorted alphabetically)."""
        return "|".join(sorted([model_a, model_b]))
    
    def get_match_count(self, model_a: str, model_b: str) -> int:
        """Get the number of games played between two models."""
        pair_key = self._get_pair_key(model_a, model_b)
        return self.match_counts.get(pair_key, 0)
    
    def get_rating(self, model_name: str) -> float:
        """Get ELO rating for a model (creates if new)."""
        if model_name not in self.ratings:
            self.ratings[model_name] = INITIAL_ELO
        return self.ratings[model_name]
    
    def update_ratings(self, model_a: str, model_b: str, score_a: float):
        """
        Update ELO ratings after a match.
        
        Args:
            model_a: First model name
            model_b: Second model name  
            score_a: Score for model_a (fraction of games won, 0.0 to 1.0)
        """
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)
        
        # Expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a
        
        # Update ratings
        self.ratings[model_a] = rating_a + K_FACTOR * (score_a - expected_a)
        self.ratings[model_b] = rating_b + K_FACTOR * ((1 - score_a) - expected_b)
    
    def record_match(self, model_a: str, model_b: str, wins_a: int, wins_b: int):
        """Record a match result."""
        total = wins_a + wins_b
        if total == 0:
            return
        
        score_a = wins_a / total
        self.update_ratings(model_a, model_b, score_a)
        
        self.matches.append({
            'model_a': model_a,
            'model_b': model_b,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'score_a': score_a,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update best model
        self._update_best()
        
        # Update match counts for matchmaking heuristics
        pair_key = self._get_pair_key(model_a, model_b)
        self.match_counts[pair_key] = self.match_counts.get(pair_key, 0) + total
        
        self.save()
    
    def _update_best(self):
        """Find and update the best model, syncing to disk."""
        best_rating = 0
        best_name = None
        for name, rating in self.ratings.items():
            if rating > best_rating:
                best_rating = rating
                best_name = name
        
        if best_name and self.best_model != best_name:
            self.best_model = best_name
            self._sync_best_model()

    def _sync_best_model(self):
        """Sync model_best.pt on disk with the current best model."""
        import shutil
        best_model_name = self.best_model
        if best_model_name:
            src_path = os.path.join(self.checkpoint_dir, best_model_name)
            dst_path = os.path.join(self.checkpoint_dir, Config.BEST_MODEL)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
    def discover_models(self):
        """Scans for new iteration checkpoints and adds them to ratings."""
        pattern = os.path.join(self.checkpoint_dir, "iteration_*.pt")
        all_models = [os.path.basename(f) for f in glob.glob(pattern)]
        
        new_found = False
        for model in all_models:
            if model not in self.ratings:
                self.ratings[model] = INITIAL_ELO
                print(f"Discovered new model: {model} (Initial ELO: {INITIAL_ELO})")
                new_found = True
        
        if new_found:
            self.save()
    

    
    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get models sorted by rating."""
        items = self.ratings.items()
        return sorted(items, key=lambda x: x[1], reverse=True)



class Arena:
    """Runs matches between models."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.state = ArenaState()
    
    def load_model(self, model_path: str) -> Tuple[AlphaZeroNet, MCTS]:
        """Load a model and create MCTS for it."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        config = checkpoint.get('config', {})
        num_blocks = config.get('num_blocks', Config.RESNET_BLOCKS)
        num_filters = config.get('num_filters', Config.RESNET_FILTERS)
        
        model = AlphaZeroNet(num_blocks=num_blocks, num_filters=num_filters).to(self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        mcts = MCTS(model, num_simulations=Config.MCTS_SIMULATIONS_INFERENCE, device=self.device)
        return model, mcts
    
    def play_game(self, mcts_white: MCTS, mcts_black: MCTS, start_game: Optional[BreakthroughGame] = None) -> int:
        """
        Play a single game between two MCTS instances.
        
        Args:
            mcts_white: MCTS instance for white player
            mcts_black: MCTS instance for black player
            start_game: Optional starting position (for paired openings)
        
        Returns:
            1 if white wins, -1 if black wins
        """
        if start_game is not None:
            game = start_game.clone()
        else:
            game = BreakthroughGame()
        
        while not game.is_terminal():
            if game.turn == WHITE:
                mcts = mcts_white
            else:
                mcts = mcts_black
            
            # Run MCTS and pick best move
            root = mcts.search(game, add_noise=False)
            
            best_action = -1
            best_visits = -1
            for action, child in root.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_action = action
            
            move = game.decode_action(best_action)
            game.step(move)
        
        w, l = game.get_result()
        # Breakthrough always has a winner - no draws possible
        assert w == 1.0 or l == 1.0, "Breakthrough game ended without a winner"
        return 1 if w == 1.0 else -1
    
    def play_match(self, model_a_path: str, model_b_path: str, num_games: int = GAMES_PER_MATCH) -> Tuple[int, int]:
        """
        Play a match between two models.
        
        Args:
            model_a_path: Path to first model
            model_b_path: Path to second model
            num_games: Number of games to play (split evenly between colors)
        
        Returns:
            Tuple of (wins_a, wins_b)
        """
        model_a, mcts_a = self.load_model(model_a_path)
        model_b, mcts_b = self.load_model(model_b_path)
        
        wins_a = 0
        wins_b = 0
        
        games_per_side = num_games // 2
        
        # Model A plays as White
        for _ in tqdm(range(games_per_side), desc=f"A as White", leave=False):
            result = self.play_game(mcts_a, mcts_b)
            if result == 1:
                wins_a += 1
            else:
                wins_b += 1
        
        # Model A plays as Black
        for _ in tqdm(range(games_per_side), desc=f"A as Black", leave=False):
            result = self.play_game(mcts_b, mcts_a)
            if result == 1:
                wins_b += 1
            else:
                wins_a += 1
        
        return wins_a, wins_b
    
    def _get_random_opening(self, num_moves: int = RANDOM_OPENING_MOVES) -> BreakthroughGame:
        """
        Generate a random opening position by playing random legal moves.
        
        Args:
            num_moves: Number of random moves to play from start position
            
        Returns:
            A game state after num_moves random moves
        """
        game = BreakthroughGame()
        for _ in range(num_moves):
            if game.is_terminal():
                break
            moves = game.get_legal_moves()
            if not moves:
                break
            move = random.choice(moves)
            game.step(move)
        return game
    
    def play_paired_match(self, mcts_a: MCTS, mcts_b: MCTS, start_game: BreakthroughGame) -> Tuple[int, int]:
        """
        Play a paired match (2 games from same position, swapping colors).
        
        This reduces opening bias by ensuring both models play both sides
        of the same opening position.
        
        Args:
            mcts_a: MCTS instance for first model
            mcts_b: MCTS instance for second model
            start_game: The starting position for both games
            
        Returns:
            Tuple of (wins_a, wins_b)
        """
        wins_a = 0
        wins_b = 0
        
        # Game 1: A as White, B as Black
        result = self.play_game(mcts_a, mcts_b, start_game)
        if result == 1:
            wins_a += 1
        else:
            wins_b += 1
        
        # Game 2: B as White, A as Black
        result = self.play_game(mcts_b, mcts_a, start_game)
        if result == 1:
            wins_b += 1
        else:
            wins_a += 1
        
        return wins_a, wins_b
    
    def select_matchup(self) -> Optional[Tuple[str, str, float]]:
        """
        Select the best pair of models for the next automated match.
        
        Uses the heuristic:
        S = (p * (1-p)) / (1 + sqrt(N)) * exp(lambda * z_top)
        
        Where:
        - p = expected win probability from ELO
        - N = number of games already played between the pair
        - z_top = z-score of the higher-rated model in the pair
        - lambda = bias strength (BIAS_LAMBDA)
        
        Returns:
            Tuple of (model_a, model_b, score) or None if < 2 models
        """
        rated_models = list(self.state.ratings.keys())
        if len(rated_models) < 2:
            return None
        
        # Calculate pool statistics for normalization
        ratings = list(self.state.ratings.values())
        mu = np.mean(ratings)
        sigma = np.std(ratings)
        epsilon = 1e-9
        
        scored_pairs = []
        for model_a, model_b in combinations(rated_models, 2):
            rating_a = self.state.get_rating(model_a)
            rating_b = self.state.get_rating(model_b)
            
            # Expected score (probability A wins)
            p = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
            
            # Base score: outcome variance (max at p=0.5)
            variance = p * (1 - p)
            
            # Match count penalty (slower decay with sqrt)
            n_games = self.state.get_match_count(model_a, model_b)
            base_score = variance / (1 + math.sqrt(n_games))
            
            # Higher-ranked bias
            r_top = max(rating_a, rating_b)
            z_top = (r_top - mu) / (sigma + epsilon)
            multiplier = math.exp(BIAS_LAMBDA * z_top)
            
            final_score = base_score * multiplier
            scored_pairs.append((model_a, model_b, final_score))
        
        # Sort by score descending
        scored_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Epsilon-greedy selection
        if random.random() < EXPLORATION_RATE and len(scored_pairs) >= TOP_K:
            # Pick randomly from top-K
            selected = random.choice(scored_pairs[:TOP_K])
        else:
            # Pick the best
            selected = scored_pairs[0]
        
        return selected
    
def run_arena():
    """
    Main arena loop.
    
    Continuously scans for new models and performs automated matchmaking
    between existing models to refine ELO ratings.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Arena using device: {device}")
    
    arena = Arena(device=str(device))
    
    print("Arena started. Continuously matchmaking...")
    print("Higher-rank bias and ELO variance will prioritize new models.")
    print("Press Ctrl+C to stop.\n")
    
    while True:
        # Scan for new models
        arena.state.discover_models()
        
        # Select best matchup from all models
        matchup = arena.select_matchup()
        
        if matchup is None:
            # Less than 2 models, wait for more
            print("Waiting for at least 2 models to start matchmaking...")
            time.sleep(30)
            continue
        
        model_a, model_b, score = matchup
        print(f"\n{'='*50}")
        print(f"MATCHMAKING: {model_a} vs {model_b}")
        print(f"Selection score: {score:.5f}")
        print(f"{'='*50}")
        
        # Generate random opening for paired match
        opening = arena._get_random_opening()
        
        # Load models
        model_a_path = os.path.join(arena.state.checkpoint_dir, model_a)
        model_b_path = os.path.join(arena.state.checkpoint_dir, model_b)
        _, mcts_a = arena.load_model(model_a_path)
        _, mcts_b = arena.load_model(model_b_path)
        
        # Play 4 games in total
        # Part 1: Standard start (2 games)
        wins_a_std, wins_b_std = arena.play_paired_match(mcts_a, mcts_b, BreakthroughGame())
        
        # Part 2: Random opening (2 games)
        wins_a_rnd, wins_b_rnd = arena.play_paired_match(mcts_a, mcts_b, opening)
        
        wins_a = wins_a_std + wins_a_rnd
        wins_b = wins_b_std + wins_b_rnd
        
        print(f"Result: {model_a} {wins_a}-{wins_b} {model_b} "
              f"({wins_a_std}-{wins_b_std} std, {wins_a_rnd}-{wins_b_rnd} rnd)")
        
        # Record match
        arena.state.record_match(model_a, model_b, wins_a, wins_b)
        
        # Print updated ratings
        print(f"Updated ratings: {model_a}={arena.state.get_rating(model_a):.0f}, "
              f"{model_b}={arena.state.get_rating(model_b):.0f}")
        
        # Print leaderboard
        print(f"\nLeaderboard:")
        for rank, (name, rating) in enumerate(arena.state.get_leaderboard()[:10], 1):
            marker = " *" if name == arena.state.best_model else ""
            print(f"  {rank}. {name}: {rating:.0f}{marker}")



if __name__ == "__main__":
    run_arena()
