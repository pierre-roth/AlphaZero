"""
Monte Carlo Tree Search with Batched Evaluation.

This module implements MCTS with:
- FPU for better exploration of unvisited nodes
- Configurable exploration constant (c_puct)
- Dirichlet noise at root for training
- Batched neural network evaluation for efficiency
- Both single-game search() and multi-game search_batch() methods
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from src.game import BreakthroughGame, NUM_ACTIONS, WHITE
from src.config import Config


class Node:
    """MCTS tree node."""
    
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children']
    
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: Dict[int, 'Node'] = {}
    
    def value(self) -> float:
        """Return mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0


class MCTS:
    """
    Monte Carlo Tree Search with batched neural network evaluation.
    
    Supports both single-game inference via search() and parallel multi-game
    execution via search_batch() for efficient training.
    
    Args:
        model: Neural network for evaluation
        num_simulations: Number of MCTS simulations per search
        c_puct: Exploration constant (higher = more exploration)
        fpu_value: Value to use for unvisited nodes (First Play Urgency)
        device: Torch device for inference
    """
    
    def __init__(
        self,
        model,
        num_simulations: int = Config.MCTS_SIMULATIONS,
        c_puct: float = Config.C_PUCT,
        fpu_reduction: float = Config.FPU_REDUCTION,
        device: str = "cpu"
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.device = device
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def search(self, game: BreakthroughGame, root: Optional[Node] = None, 
               add_noise: bool = False) -> Node:
        """
        Run MCTS for a single game.
        
        This is a convenience wrapper around search_batch() for single-game use.
        
        Args:
            game: Current game state
            root: Optional existing root node (for tree reuse)
            add_noise: Whether to add Dirichlet noise at root
            
        Returns:
            Root node after search
        """
        roots = [root] if root is not None else None
        return self.search_batch([game], roots=roots, add_noise=add_noise)[0]
    
    def search_batch(
        self,
        games: List[BreakthroughGame],
        roots: Optional[List[Node]] = None,
        add_noise: bool = False
    ) -> List[Node]:
        """
        Run MCTS for multiple games with batched neural network evaluation.
        
        This is the core method that efficiently runs MCTS across multiple
        games simultaneously, batching all neural network calls.
        
        Args:
            games: List of game states
            roots: Optional list of existing root nodes (for tree reuse)
            add_noise: Whether to add Dirichlet noise at root
            
        Returns:
            List of root nodes after search
        """
        num_games = len(games)
        
        # Initialize or reuse roots
        if roots is None:
            roots = [Node(0) for _ in range(num_games)]
        else:
            # Ensure we have the right number of roots and replace None with new nodes
            while len(roots) < num_games:
                roots.append(None)
            for i in range(num_games):
                if roots[i] is None:
                    roots[i] = Node(0)
        
        # Expand roots that need expansion
        games_to_expand = []
        indices_to_expand = []
        for i, (root, game) in enumerate(zip(roots, games)):
            if not root.is_expanded():
                games_to_expand.append(game)
                indices_to_expand.append(i)
        
        if games_to_expand:
            results = self._batch_evaluate(games_to_expand)
            for idx, (policy_probs, value) in zip(indices_to_expand, results):
                self._expand_node(roots[idx], games[idx], policy_probs)
        
        # Add Dirichlet noise if training
        if add_noise:
            for root in roots:
                if root.is_expanded():
                    self._add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(self.num_simulations):
            leaf_games = []
            leaf_indices = []
            search_paths = []
            scratch_games = []
            
            for i in range(num_games):
                node = roots[i]
                scratch_game = games[i].clone()
                path = [node]
                
                # Selection: traverse until leaf
                while node.is_expanded():
                    action, node = self._select_child(node)
                    move = scratch_game.decode_action(action)
                    scratch_game.step(move)
                    path.append(node)
                
                search_paths.append(path)
                scratch_games.append(scratch_game)
                
                # If not terminal, we need to expand
                if not scratch_game.is_terminal():
                    leaf_games.append(scratch_game)
                    leaf_indices.append(i)
            
            # Batch evaluate all leaf nodes
            if leaf_games:
                results = self._batch_evaluate(leaf_games)
                
                for idx, (policy_probs, value) in zip(leaf_indices, results):
                    leaf_node = search_paths[idx][-1]
                    self._expand_node(leaf_node, scratch_games[idx], policy_probs)
                    self._backpropagate(search_paths[idx], value)
            
            # Handle terminal nodes
            for i in range(num_games):
                if i not in leaf_indices:
                    scratch_game = scratch_games[i]
                    value = self._get_terminal_value(scratch_game)
                    self._backpropagate(search_paths[i], value)
        
        return roots
    
    def get_action_probs(self, root: Node, temperature: float = 1.0) -> np.ndarray:
        """
        Convert root visit counts to action probabilities.
        
        Args:
            root: Root node after search
            temperature: Temperature for visit count distribution.
                         0 = select best, 1 = proportional, >1 = more random
        
        Returns:
            Array of shape (192,) with action probabilities
        """
        probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        
        for action, child in root.children.items():
            probs[action] = child.visit_count
        
        if temperature == 0:
            # Deterministic: select best
            best_action = int(np.argmax(probs))
            probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            probs[best_action] = 1.0
        else:
            # Apply temperature
            probs = np.power(probs, 1.0 / temperature)
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                # Fallback to uniform if no visits
                legal_actions = list(root.children.keys())
                if legal_actions:
                    probs[legal_actions] = 1.0 / len(legal_actions)
        
        return probs
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _batch_evaluate(self, games: List[BreakthroughGame]) -> List[Tuple[np.ndarray, float]]:
        """
        Evaluate multiple game states in a single batch.
        
        Args:
            games: List of game states to evaluate
            
        Returns:
            List of (policy_probs, value) tuples for each game
        """
        if not games:
            return []
        
        # Encode all states
        states = np.array([g.get_encoded_state() for g in games])
        batch = torch.tensor(states, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, wl_logits = self.model(batch)
        
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
        wl_probs = torch.softmax(wl_logits, dim=1).cpu().numpy()
        
        # Convert WL to scalar values: P(win) - P(loss)
        values = wl_probs[:, 0] - wl_probs[:, 1]
        
        return list(zip(policy_probs, values))
    
    def _expand_node(self, node: Node, game: BreakthroughGame, policy_probs: np.ndarray):
        """Expand a node with the given policy probabilities."""
        legal_moves = game.get_legal_moves()
        
        prior_sum = 0.0
        legal_actions = []
        
        for move in legal_moves:
            action = game.encode_action(move)
            if action < len(policy_probs):
                prior_sum += policy_probs[action]
                legal_actions.append((action, policy_probs[action]))
        
        # Create child nodes with normalized priors
        for action, prior in legal_actions:
            if prior_sum > 0:
                normalized_prior = prior / prior_sum
            else:
                normalized_prior = 1.0 / len(legal_actions) if legal_actions else 1.0
            node.children[action] = Node(normalized_prior)
    
    def _select_child(self, node: Node) -> Tuple[int, Node]:
        """Select child with highest PUCT score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        # Use max(1, ...) to avoid sqrt(0) which would ignore policy priors on first visit
        sqrt_parent = math.sqrt(max(1, node.visit_count))
        
        # Compute parent Q for FPU (from current player's perspective)
        parent_q = node.value() if node.visit_count > 0 else 0.0
        
        for action, child in node.children.items():
            # Use FPU for unvisited nodes: parent_Q - fpu_reduction
            # This prevents "despair exploration" in losing positions
            if child.visit_count == 0:
                q = parent_q - self.fpu_reduction
            else:
                # Q from child's perspective (opponent), so negate
                q = -child.value()
            
            # U: exploration bonus
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _add_dirichlet_noise(self, node: Node):
        """Add Dirichlet noise to root node priors for exploration."""
        alpha = Config.DIRICHLET_ALPHA
        epsilon = Config.DIRICHLET_EPSILON
        
        actions = list(node.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
    
    def _get_terminal_value(self, game: BreakthroughGame) -> float:
        """Get value of terminal state from current player's perspective."""
        w, l = game.get_result()
        # Result is from WHITE's perspective, convert to current player
        white_value = w - l
        if game.turn == WHITE:
            return white_value
        return -white_value
    
    def _backpropagate(self, path: List[Node], value: float):
        """Backpropagate value through the search path."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Switch perspective


if __name__ == "__main__":
    # Quick test
    from src.model import AlphaZeroNet
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and MCTS
    model = AlphaZeroNet().to(device)
    mcts = MCTS(model, num_simulations=50, device=str(device))
    
    # Test single-game search
    print("\nTesting single-game search...")
    game = BreakthroughGame()
    root = mcts.search(game, add_noise=True)
    print(f"Root visit count: {root.visit_count}")
    print(f"Number of children: {len(root.children)}")
    
    # Test batch search
    print("\nTesting batch search...")
    games = [BreakthroughGame() for _ in range(4)]
    roots = mcts.search_batch(games, add_noise=True)
    print(f"Searched {len(roots)} games")
    for i, root in enumerate(roots):
        print(f"  Game {i}: {root.visit_count} visits, {len(root.children)} children")
    
    # Get action probs
    probs = mcts.get_action_probs(roots[0], temperature=1.0)
    print(f"\nAction probs sum: {probs.sum():.4f}")
    print(f"Top action: {np.argmax(probs)} with prob {probs.max():.4f}")
    
    print("\nMCTS test passed!")
