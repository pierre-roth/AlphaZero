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
BIAS_LAMBDA = 0.05  # Strength of higher-ranked bias
RANDOM_OPENING_MOVES = 6  # Number of random moves for paired openings


class ArenaState:
    """Persistent state for the arena."""
    
    def __init__(self, checkpoint_dir: str = Config.CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, "arena_state.json")
        
        self.ratings: Dict[str, float] = {}
        self.matches: List[dict] = []
        # Track best model per size
        self.best_models: Dict[str, str] = {}  # size -> model_name
        # Track cross-size matches (best vs best of different sizes)
        self.cross_size_matches: List[dict] = []
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
                self.best_models = data.get('best_models', {})
                self.cross_size_matches = data.get('cross_size_matches', [])
                # Rebuild match_counts from historical data for consistency
                self._rebuild_match_counts()
            print(f"Loaded arena state: {len(self.ratings)} models rated, best per size: {self.best_models}")
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
        # Count from cross-size matches
        for match in self.cross_size_matches:
            pair_key = self._get_pair_key(match['model_a'], match['model_b'])
            total_games = match['wins_a'] + match['wins_b']
            self.match_counts[pair_key] = self.match_counts.get(pair_key, 0) + total_games

    
    def save(self):
        """Save state to disk."""
        data = {
            'ratings': self.ratings,
            'matches': self.matches,
            'best_models': self.best_models,
            'cross_size_matches': self.cross_size_matches,
            'match_counts': self.match_counts,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def _get_model_size(model_name: str) -> Optional[str]:
        """Extract model size from checkpoint name (e.g., 'iteration_5_medium.pt' -> 'medium')."""
        import re
        match = re.search(r'_(small|medium|large)\.pt$', model_name)
        return match.group(1) if match else None
    
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
        
        # Update best model per size for both participants
        for model in [model_a, model_b]:
            size = self._get_model_size(model)
            if size:
                self._update_best_for_size(size)
        
        # Update match counts for matchmaking heuristics
        pair_key = self._get_pair_key(model_a, model_b)
        self.match_counts[pair_key] = self.match_counts.get(pair_key, 0) + total
        
        self.save()
    
    def _update_best_for_size(self, size: str):
        """Find and update the best model for a given size, syncing to disk."""
        best_rating = 0
        best_for_size = None
        for name, rating in self.ratings.items():
            if self._get_model_size(name) == size and rating > best_rating:
                best_rating = rating
                best_for_size = name
        
        if best_for_size and self.best_models.get(size) != best_for_size:
            self.best_models[size] = best_for_size
            self._sync_best_model(size)
    
    def _sync_best_model(self, size: str):
        """Sync model_best_{size}.pt on disk with the current best model."""
        import shutil
        best_model_name = self.best_models.get(size)
        if best_model_name:
            src_path = os.path.join(self.checkpoint_dir, best_model_name)
            dst_path = os.path.join(self.checkpoint_dir, f"model_best_{size}.pt")
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"Synced model_best_{size}.pt -> {best_model_name}")

    
    def get_unevaluated_models(self) -> List[str]:
        """Find iteration checkpoints that haven't been evaluated yet.
        
        Excludes models that are currently set as best_models (baselines),
        since they have no opponent to play against yet.
        """
        pattern = os.path.join(self.checkpoint_dir, "iteration_*_*.pt")
        all_models = [os.path.basename(f) for f in glob.glob(pattern)]
        
        evaluated = set()
        for match in self.matches:
            evaluated.add(match['model_a'])
            evaluated.add(match['model_b'])
        
        # Exclude current best_models (baselines with no opponent)
        baseline_models = set(self.best_models.values())
        
        return [m for m in all_models if m not in evaluated and m not in baseline_models]

    
    def get_best_for_size(self, size: str) -> Optional[str]:
        """Get the best model for a specific size."""
        return self.best_models.get(size)
    
    def get_leaderboard(self, size: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get models sorted by rating, optionally filtered by size."""
        items = self.ratings.items()
        if size:
            items = [(n, r) for n, r in items if self._get_model_size(n) == size]
        return sorted(items, key=lambda x: x[1], reverse=True)
    
    def have_cross_size_match(self, model_a: str, model_b: str) -> bool:
        """Check if two models have already played a cross-size match."""
        for match in self.cross_size_matches:
            if (match['model_a'] == model_a and match['model_b'] == model_b) or \
               (match['model_a'] == model_b and match['model_b'] == model_a):
                return True
        return False
    
    def record_cross_size_match(self, model_a: str, model_b: str, wins_a: int, wins_b: int):
        """Record a cross-size match result and update ratings."""
        size_a = self._get_model_size(model_a)
        size_b = self._get_model_size(model_b)
        
        total = wins_a + wins_b
        if total > 0:
            # Update ELO ratings
            score_a = wins_a / total
            self.update_ratings(model_a, model_b, score_a)
            
            # Update match counts for matchmaking heuristics
            pair_key = self._get_pair_key(model_a, model_b)
            self.match_counts[pair_key] = self.match_counts.get(pair_key, 0) + total
        
        self.cross_size_matches.append({
            'model_a': model_a,
            'model_b': model_b,
            'size_a': size_a,
            'size_b': size_b,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'timestamp': datetime.now().isoformat()
        })
        self.save()



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
    
    def evaluate_model(self, model_name: str):
        """
        Evaluate a new model against the current best of the same size.
        
        Uses hybrid evaluation strategy:
        - 50% games from standard start position
        - 50% games from random paired openings (reduces opening bias)
        """
        model_path = os.path.join(self.state.checkpoint_dir, model_name)
        
        # Extract model size
        size = ArenaState._get_model_size(model_name)
        if not size:
            print(f"Could not determine size for {model_name}, skipping")
            return
        
        # Get best model for this size
        best_for_size = self.state.get_best_for_size(size)
        
        if best_for_size:
            opponent_name = best_for_size
            opponent_path = os.path.join(self.state.checkpoint_dir, opponent_name)
        else:
            # No best model yet for this size, use the new model as baseline
            self.state.ratings[model_name] = INITIAL_ELO
            self.state.best_models[size] = model_name
            self.state.save()
            print(f"First {size} model {model_name} set as baseline (ELO: {INITIAL_ELO})")
            
            # Copy as best model for this size
            best_path = os.path.join(self.state.checkpoint_dir, f"model_best_{size}.pt")
            import shutil
            shutil.copy(model_path, best_path)
            print(f"Saved as model_best_{size}.pt")
            return
        
        print(f"\nEvaluating {model_name} vs {opponent_name} ({size})...")
        print("Using hybrid evaluation: 50% standard, 50% random paired openings")
        
        # Load models
        _, mcts_new = self.load_model(model_path)
        _, mcts_old = self.load_model(opponent_path)
        
        wins_new = 0
        wins_old = 0
        
        # Part 1: Standard start (10 games = 5 per side)
        print("Part 1: Standard start games...")
        for _ in tqdm(range(5), desc="New as White", leave=False):
            result = self.play_game(mcts_new, mcts_old)
            if result == 1:
                wins_new += 1
            else:
                wins_old += 1
        
        for _ in tqdm(range(5), desc="New as Black", leave=False):
            result = self.play_game(mcts_old, mcts_new)
            if result == 1:
                wins_old += 1
            else:
                wins_new += 1
        
        # Part 2: Random paired openings (10 games = 5 openings × 2 games each)
        print("Part 2: Random paired opening games...")
        for i in tqdm(range(5), desc="Paired openings", leave=False):
            opening = self._get_random_opening()
            w_new, w_old = self.play_paired_match(mcts_new, mcts_old, opening)
            wins_new += w_new
            wins_old += w_old
        
        print(f"Result: {model_name} {wins_new}-{wins_old} {opponent_name}")
        
        # Record match
        self.state.record_match(model_name, opponent_name, wins_new, wins_old)
        
        # Check if new model is now best for its size
        if self.state.best_models.get(size) == model_name:
            best_path = os.path.join(self.state.checkpoint_dir, f"model_best_{size}.pt")
            import shutil
            shutil.copy(model_path, best_path)
            print(f"New best {size} model! ELO: {self.state.ratings[model_name]:.0f}")
        
        # Print leaderboard for this size
        print(f"\nLeaderboard ({size}):")
        for rank, (name, rating) in enumerate(self.state.get_leaderboard(size)[:10], 1):
            marker = " *" if name == self.state.best_models.get(size) else ""
            print(f"  {rank}. {name}: {rating:.0f}{marker}")


def run_arena():
    """
    Main arena loop.
    
    Continuously scans for new models and evaluates them.
    When no new models are found, performs automated matchmaking
    between existing models to refine ELO ratings.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Arena using device: {device}")
    
    arena = Arena(device=str(device))
    
    print("Arena started. Scanning for new models...")
    print("(Run this alongside training to evaluate models as they're generated)")
    print("Automated matchmaking will run when no new models are available.")
    print("Press Ctrl+C to stop.\n")
    
    while True:
        # Find unevaluated models
        new_models = arena.state.get_unevaluated_models()
        
        if new_models:
            # Sort by iteration number
            new_models.sort(key=lambda x: int(x.replace("iteration_", "").replace(".pt", "").replace("_large", "").replace("_medium", "").replace("_small", "")))
            
            for model_name in new_models:
                print(f"\n{'='*50}")
                arena.evaluate_model(model_name)
                print(f"{'='*50}")
        else:
            # No new models - run automated matchmaking
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
            
            # Play paired match (2 games from same opening)
            wins_a, wins_b = arena.play_paired_match(mcts_a, mcts_b, opening)
            
            print(f"Result: {model_a} {wins_a}-{wins_b} {model_b}")
            
            # Record match
            arena.state.record_match(model_a, model_b, wins_a, wins_b)
            
            # Print updated ratings
            print(f"Updated ratings: {model_a}={arena.state.get_rating(model_a):.0f}, "
                  f"{model_b}={arena.state.get_rating(model_b):.0f}")


def run_cross_size_arena():
    """
    Cross-size arena: pit best models of different sizes against each other.
    
    Only runs matches that haven't been played yet (tracks by exact model names).
    Results are saved to arena_state.json under 'cross_size_matches'.
    """
    from itertools import combinations
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Cross-size arena using device: {device}")
    
    arena = Arena(device=str(device))
    
    # Get all sizes that have a best model
    available_sizes = list(arena.state.best_models.keys())
    print(f"\nAvailable sizes with best models: {available_sizes}")
    
    if len(available_sizes) < 2:
        print("Need at least 2 different model sizes to run cross-size arena.")
        print("Train models of different sizes first.")
        return
    
    # Show current best models
    print("\nCurrent best models:")
    for size, model_name in arena.state.best_models.items():
        rating = arena.state.get_rating(model_name)
        print(f"  {size}: {model_name} (ELO: {rating:.0f})")
    
    # Generate all pairs
    size_pairs = list(combinations(available_sizes, 2))
    print(f"\nChecking {len(size_pairs)} size pairings...")
    
    matches_played = 0
    for size_a, size_b in size_pairs:
        model_a = arena.state.best_models[size_a]
        model_b = arena.state.best_models[size_b]
        
        # Check if this exact match has been played
        if arena.state.have_cross_size_match(model_a, model_b):
            print(f"\n[SKIP] {model_a} vs {model_b} (already played)")
            continue
        
        print(f"\n{'='*60}")
        print(f"CROSS-SIZE MATCH: {size_a.upper()} vs {size_b.upper()}")
        print(f"  {model_a}")
        print(f"  vs")
        print(f"  {model_b}")
        print(f"{'='*60}")
        
        # Load models and play
        model_a_path = os.path.join(arena.state.checkpoint_dir, model_a)
        model_b_path = os.path.join(arena.state.checkpoint_dir, model_b)
        
        wins_a, wins_b = arena.play_match(model_a_path, model_b_path)
        
        # Record result
        arena.state.record_cross_size_match(model_a, model_b, wins_a, wins_b)
        
        # Determine winner
        if wins_a > wins_b:
            winner = f"{size_a.upper()} ({model_a})"
        elif wins_b > wins_a:
            winner = f"{size_b.upper()} ({model_b})"
        else:
            winner = "TIE"
        
        print(f"\nResult: {model_a} {wins_a}-{wins_b} {model_b}")
        print(f"Winner: {winner}")
        matches_played += 1
    
    print(f"\n{'='*60}")
    print(f"Cross-size arena complete. Played {matches_played} new matches.")
    
    # Show cross-size leaderboard
    if arena.state.cross_size_matches:
        print("\nCross-size match history:")
        for match in arena.state.cross_size_matches:
            result = f"{match['wins_a']}-{match['wins_b']}"
            print(f"  {match['size_a']} vs {match['size_b']}: {result}")
            print(f"    {match['model_a']} vs {match['model_b']}")


if __name__ == "__main__":
    run_arena()
