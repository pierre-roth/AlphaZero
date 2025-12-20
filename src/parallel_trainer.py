"""
Parallel Self-Play Trainer with Tree Reuse.

This module implements parallel game execution where multiple games run simultaneously,
with neural network evaluations batched together for efficiency on GPU/MPS.

Key features:
- Tree reuse: preserves subtree after each move
- Batched neural network evaluation via unified MCTS
- WL (Win/Loss) training targets
- Data augmentation via horizontal mirroring
- Cosine annealing learning rate schedule
"""

import os
import re
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple

from src.game import BreakthroughGame, NUM_ACTIONS, WHITE, BOARD_SIZE
from src.model import AlphaZeroNet
from src.config import Config
from src.mcts import MCTS, Node


def augment_example(state: np.ndarray, policy: np.ndarray, wl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a horizontally mirrored version of a training example.
    
    Breakthrough has horizontal symmetry - mirroring left/right produces
    an equally valid position, effectively doubling our training data.
    
    Args:
        state: Board state of shape (3, 8, 8)
        policy: Policy distribution of shape (192,)
        wl: Win/Loss target of shape (2,)
    
    Returns:
        Mirrored (state, policy, wl) tuple
    """
    # Mirror the board state horizontally (flip columns)
    mirrored_state = np.flip(state, axis=2).copy()
    
    # Mirror the policy: for each action, swap left and right diagonals
    # Action encoding: square * 3 + direction
    # Directions: 0=forward, 1=diag-left, 2=diag-right
    mirrored_policy = np.zeros_like(policy)
    
    for square in range(64):
        row = square // BOARD_SIZE
        col = square % BOARD_SIZE
        mirrored_col = BOARD_SIZE - 1 - col
        mirrored_square = row * BOARD_SIZE + mirrored_col
        
        # Forward move stays forward
        mirrored_policy[mirrored_square * 3 + 0] = policy[square * 3 + 0]
        # Left diagonal becomes right diagonal
        mirrored_policy[mirrored_square * 3 + 2] = policy[square * 3 + 1]
        # Right diagonal becomes left diagonal
        mirrored_policy[mirrored_square * 3 + 1] = policy[square * 3 + 2]
    
    # WL target stays the same (winning position is still winning when mirrored)
    return mirrored_state, mirrored_policy, wl


class GameDataset(Dataset):
    """Dataset for training examples with on-the-fly augmentation."""
    
    def __init__(self, examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], augment: bool = True):
        self.examples = examples
        self.augment = augment
    
    def __len__(self):
        # If augmenting, we have 2x the examples (original + mirrored)
        return len(self.examples) * 2 if self.augment else len(self.examples)
    
    def __getitem__(self, idx):
        # Determine if this index is for an augmented example
        if self.augment and idx >= len(self.examples):
            # This is an augmented example
            real_idx = idx - len(self.examples)
            state, policy, wl = self.examples[real_idx]
            state, policy, wl = augment_example(state, policy, wl)
        else:
            state, policy, wl = self.examples[idx]
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(policy, dtype=torch.float32),
            torch.tensor(wl, dtype=torch.float32)
        )


class ParallelTrainer:
    """
    Trainer with parallel self-play and tree reuse.
    
    Args:
        model: Neural network to train
        device: Torch device
        num_parallel_games: Number of games to run in parallel
        num_simulations: MCTS simulations per move
        model_size: Size name for checkpoint naming ('small', 'medium', 'large')
    """
    
    def __init__(
        self,
        model: AlphaZeroNet,
        device: str = "cpu",
        num_parallel_games: int = Config.PARALLEL_GAMES,
        num_simulations: int = Config.MCTS_SIMULATIONS,
        checkpoint_dir: str = Config.CHECKPOINT_DIR,
    ):
        self.model = model
        self.device = device
        self.num_parallel_games = num_parallel_games
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        # Cosine annealing LR scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=Config.LR_SCHEDULER_T_MAX, eta_min=Config.LR_SCHEDULER_ETA_MIN)
        self.mcts = MCTS(model, num_simulations=num_simulations, device=device)
        self.examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.max_examples = Config.BUFFER_SIZE  # Cap replay buffer at this size
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def execute_parallel_episodes(self, num_games: Optional[int] = None) -> List[Tuple]:
        """
        Run multiple games in parallel with tree reuse.
        
        Returns:
            List of (state, policy, wl) training examples
        """
        if num_games is None:
            num_games = self.num_parallel_games
        
        # Initialize games and tracking
        games = [BreakthroughGame() for _ in range(num_games)]
        roots: List[Optional[Node]] = [None] * num_games  # Tree reuse
        game_data = [[] for _ in range(num_games)]  # (state, probs, turn) per game
        step_counts = [0] * num_games
        active_games = list(range(num_games))
        
        all_examples = []
        
        pbar = tqdm(desc=f"Parallel Self-Play ({num_games} games)", unit="step")
        
        while active_games:
            # Get active game states and roots
            active_game_objs = [games[i] for i in active_games]
            active_roots = [roots[i] for i in active_games]
            
            # Determine temperatures (1.0 for first N moves, then 0)
            temps = [1.0 if step_counts[i] < Config.TEMPERATURE_THRESHOLD else 0.0 
                     for i in active_games]
            
            # Run batched MCTS with tree reuse
            new_roots = self.mcts.search_batch(active_game_objs, active_roots, add_noise=True)
            
            # Update roots for active games
            for j, i in enumerate(active_games):
                roots[i] = new_roots[j]
            
            # Process results
            finished = []
            for j, i in enumerate(active_games):
                game = games[i]
                root = roots[i]
                temp = temps[j]
                
                # Get action probabilities
                probs = self.mcts.get_action_probs(root, temperature=temp)
                
                # Store training data
                state = game.get_encoded_state()
                game_data[i].append((state, probs, game.turn))
                
                # Choose and play action
                action_idx = int(np.random.choice(len(probs), p=probs))
                move = game.decode_action(action_idx)
                game.step(move)
                step_counts[i] += 1
                
                # Tree reuse: move to the subtree of the chosen action
                if action_idx in root.children:
                    roots[i] = root.children[action_idx]
                else:
                    roots[i] = None  # Reset if action not in tree
                
                # Check if game ended
                if game.is_terminal():
                    w, l = game.get_result()
                    
                    # Convert game data to training examples
                    for s, p, turn in game_data[i]:
                        # WL from the perspective of the player who made the move
                        if turn == WHITE:
                            wl = np.array([w, l], dtype=np.float32)
                        else:
                            wl = np.array([l, w], dtype=np.float32)
                        all_examples.append((s, p, wl))
                    
                    finished.append(i)
            
            # Remove finished games
            for i in finished:
                active_games.remove(i)
            
            pbar.update(1)
            pbar.set_postfix({"active": len(active_games), "examples": len(all_examples)})
        
        pbar.close()
        return all_examples
    
    def learn(self, epochs: int = 10, batch_size: int = 64):
        """Train the model on stored examples with data augmentation."""
        if not self.examples:
            return
        
        # Create dataset with augmentation (doubles the data via horizontal mirroring)
        dataset = GameDataset(self.examples, augment=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Training on {len(dataset)} examples (augmented), LR: {current_lr:.6f}")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            count = 0
            
            pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{epochs}", unit="batch")
            for states, pis, wls in pbar:
                states = states.to(self.device)
                target_pis = pis.to(self.device)
                target_wls = wls.to(self.device)
                
                # Forward pass
                pred_pis, pred_wls = self.model(states)
                
                # Policy loss (cross-entropy)
                log_probs = F.log_softmax(pred_pis, dim=1)
                loss_pi = -torch.sum(target_pis * log_probs) / target_pis.size(0)
                
                # Value loss (cross-entropy for WL)
                loss_wl = F.cross_entropy(pred_wls, target_wls)
                
                loss = loss_pi + loss_wl
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                if Config.GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_NORM)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += loss_pi.item()
                total_value_loss += loss_wl.item()
                count += 1
                
                pbar.set_postfix({
                    "loss": total_loss / count,
                    "pi": total_policy_loss / count,
                    "wl": total_value_loss / count
                })
        
        # Step the learning rate scheduler after each training call
        self.scheduler.step()
    
    # =========================================================================
    # Checkpoint Management (Model-Only)
    # =========================================================================
    
    def save_iteration_checkpoint(self, iteration: int):
        """
        Save a model-only checkpoint for the given iteration.
        
        The checkpoint contains only model state, optimizer, scheduler, and iteration.
        Training data is saved separately via append_training_data().
        Filename includes model size: iteration_N_size.pt
        """
        filename = f"iteration_{iteration}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iteration': iteration,
            'config': {
                'num_blocks': self.model.num_blocks,
                'num_filters': self.model.num_filters,
            }
        }, path)
        print(f"Checkpoint saved: {filename}")
    
    def load_iteration_checkpoint(self, iteration: int) -> bool:
        """
        Load a model checkpoint for the given iteration.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        filename = f"iteration_{iteration}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(path):
            print(f"No checkpoint found: {filename}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            # Ensure the current code's T_max is used, regardless of what was in the checkpoint
            self.scheduler.T_max = Config.LR_SCHEDULER_T_MAX
        print(f"Loaded checkpoint: {filename}")
        return True
    
    def get_latest_iteration(self) -> int:
        """
        Find the highest iteration number from existing checkpoints for this model size.
        
        Returns:
            The latest iteration number, or 0 if no checkpoints exist.
        """
        # Only look for checkpoints
        pattern = os.path.join(self.checkpoint_dir, "iteration_*.pt")
        files = glob.glob(pattern)
        
        if not files:
            return 0
        
        iterations = []
        for f in files:
            match = re.search(r'iteration_(\d+)\.pt$', f)
            if match:
                iterations.append(int(match.group(1)))
        
        return max(iterations) if iterations else 0
    
    # =========================================================================
    # Training Data Management (Separate File)
    # =========================================================================
    
    def _get_data_file_path(self) -> str:
        """Get path to the training data file."""
        return os.path.join(self.checkpoint_dir, Config.DATA_FILE)
    
    def append_training_data(self, new_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """
        Append new training examples to the data file on disk.
        
        This method appends to the existing file (or creates a new one).
        All examples are preserved on disk indefinitely.
        """
        if not new_examples:
            return
        
        data_path = self._get_data_file_path()
        
        # Separate into arrays
        new_states = np.array([e[0] for e in new_examples])
        new_policies = np.array([e[1] for e in new_examples])
        new_wls = np.array([e[2] for e in new_examples])
        
        if os.path.exists(data_path):
            # Load existing data and append
            existing = np.load(data_path)
            states = np.concatenate([existing['states'], new_states], axis=0)
            policies = np.concatenate([existing['policies'], new_policies], axis=0)
            wls = np.concatenate([existing['wls'], new_wls], axis=0)
        else:
            states = new_states
            policies = new_policies
            wls = new_wls
        
        np.savez(data_path, states=states, policies=policies, wls=wls)
        print(f"Training data saved: {len(states)} total examples ({len(new_examples)} new)")
    
    def load_training_data(self, max_examples: Optional[int] = None) -> int:
        """
        Load training examples from the data file into memory.
        
        Args:
            max_examples: Maximum number of examples to load (most recent).
                         If None, uses self.max_examples.
        
        Returns:
            Number of examples loaded.
        """
        if max_examples is None:
            max_examples = self.max_examples
        
        data_path = self._get_data_file_path()
        
        if not os.path.exists(data_path):
            print("No training data file found")
            self.examples = []
            return 0
        
        # Use memory mapping to avoid loading the entire file into RAM.
        # This is critical as the training data file grows unbounded.
        data = np.load(data_path, mmap_mode='r')
        total_examples = len(data['states'])
        
        # Determine slice range for the most recent examples
        start_idx = max(0, total_examples - max_examples)
        
        # Slice and copy to break reference to the mmap (prevents memory leaks)
        states = data['states'][start_idx:].copy()
        policies = data['policies'][start_idx:].copy()
        wls = data['wls'][start_idx:].copy()
        
        # Convert to list of tuples for compatibility
        self.examples = [(states[i], policies[i], wls[i]) for i in range(len(states))]
        print(f"Loaded {len(self.examples)} training examples (of {total_examples} total on disk)")
        return len(self.examples)
    
    def add_examples(self, new_examples: List[Tuple]):
        """Add examples to the in-memory replay buffer, maintaining the size limit."""
        self.examples.extend(new_examples)
        # Keep only the most recent examples if we exceed the limit
        if len(self.examples) > self.max_examples:
            self.examples = self.examples[-self.max_examples:]
    
    # =========================================================================
    # Legacy Compatibility (for migration)
    # =========================================================================
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Legacy save method - saves model only, no examples."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': {
                'num_blocks': self.model.num_blocks,
                'num_filters': self.model.num_filters,
            }
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str = "checkpoint.pt"):
        """Legacy load method - for backward compatibility."""
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                # Ensure the current code's T_max is used
                self.scheduler.T_max = Config.LR_SCHEDULER_T_MAX
            # Don't load examples from legacy checkpoints during normal operation
            print(f"Checkpoint loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")


if __name__ == "__main__":
    # Quick test
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AlphaZeroNet().to(device)
    trainer = ParallelTrainer(model, device=str(device), num_parallel_games=2, num_simulations=10)
    
    print("Running 2 parallel games with 10 sims each...")
    examples = trainer.execute_parallel_episodes(num_games=2)
    print(f"Generated {len(examples)} training examples")
    
    if examples:
        s, p, w = examples[0]
        print(f"Example state shape: {s.shape}")
        print(f"Example policy shape: {p.shape}")
        print(f"Example WL: {w}")
        
        # Test augmentation
        aug_s, aug_p, aug_w = augment_example(s, p, w)
        print(f"Augmented state shape: {aug_s.shape}")
        print(f"Policy sum before/after: {p.sum():.4f} / {aug_p.sum():.4f}")
    
    print("Trainer test passed!")
