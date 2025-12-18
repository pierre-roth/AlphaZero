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
            print(f"Loaded arena state: {len(self.ratings)} models rated, best per size: {self.best_models}")
        else:
            print("No existing arena state found, starting fresh")
    
    def save(self):
        """Save state to disk."""
        data = {
            'ratings': self.ratings,
            'matches': self.matches,
            'best_models': self.best_models,
            'cross_size_matches': self.cross_size_matches,
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
        
        # Update best model per size
        size = self._get_model_size(model_a)
        if size:
            # Find best model for this size
            best_rating = 0
            best_for_size = None
            for name, rating in self.ratings.items():
                if self._get_model_size(name) == size and rating > best_rating:
                    best_rating = rating
                    best_for_size = name
            if best_for_size:
                self.best_models[size] = best_for_size
        
        self.save()
    
    def get_unevaluated_models(self) -> List[str]:
        """Find iteration checkpoints that haven't been evaluated yet."""
        pattern = os.path.join(self.checkpoint_dir, "iteration_*_*.pt")
        all_models = [os.path.basename(f) for f in glob.glob(pattern)]
        
        evaluated = set()
        for match in self.matches:
            evaluated.add(match['model_a'])
            evaluated.add(match['model_b'])
        
        return [m for m in all_models if m not in evaluated]
    
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
        """Record a cross-size match result."""
        size_a = self._get_model_size(model_a)
        size_b = self._get_model_size(model_b)
        
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
    
    def play_game(self, mcts_white: MCTS, mcts_black: MCTS) -> int:
        """
        Play a single game between two MCTS instances.
        
        Returns:
            1 if white wins, -1 if black wins
        """
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
    
    def evaluate_model(self, model_name: str):
        """Evaluate a new model against the current best of the same size (or baseline)."""
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
        wins_new, wins_old = self.play_match(model_path, opponent_path)
        
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
    Can run in parallel with training.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Arena using device: {device}")
    
    arena = Arena(device=str(device))
    
    print("Arena started. Scanning for new models...")
    print("(Run this alongside training to evaluate models as they're generated)")
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
            # No new models, wait and check again
            print(".", end="", flush=True)
            time.sleep(30)  # Check every 30 seconds


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
