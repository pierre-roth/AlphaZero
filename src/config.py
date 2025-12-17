"""
Centralized Configuration for AlphaZero Breakthrough.

All hyperparameters, architecture settings, and paths should be defined here.
Other modules should import from this file rather than hardcoding values.
"""

import os


class Config:
    # ==========================================================================
    # Game Constants (Breakthrough)
    # ==========================================================================
    BOARD_SIZE = 8              # 8x8 board
    NUM_ACTIONS = 192           # 64 squares × 3 directions (forward, diag-left, diag-right)
    OUTPUT_ACTIONS = 192        # Alias for model.py compatibility
    INPUT_PLANES = 3            # My pieces, opponent pieces, ones

    # ==========================================================================
    # Model Architecture
    # ==========================================================================
    RESNET_BLOCKS = 6           # Number of residual blocks (smaller for simpler game)
    RESNET_FILTERS = 128        # Number of filters per conv layer
    SE_RATIO = 8                # Squeeze-Excitation reduction ratio

    # ==========================================================================
    # MCTS Parameters
    # ==========================================================================
    MCTS_SIMULATIONS = 800      # Simulations per search (training)
    MCTS_SIMULATIONS_INFERENCE = 200  # Simulations for web inference
    C_PUCT = 1.5                # Exploration constant
    # First Play Urgency: absolute value used for unvisited nodes (not relative to parent).
    # A positive value encourages exploration of untried moves.
    FPU_VALUE = 0.33
    DIRICHLET_ALPHA = 0.35       # Dirichlet noise alpha (more exploration)
    DIRICHLET_EPSILON = 0.25    # Dirichlet noise weight
    TEMPERATURE_THRESHOLD = 15  # Moves before switching to deterministic

    # ==========================================================================
    # Training Parameters
    # ==========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PARALLEL_GAMES = 32         # Games to run in parallel during self-play
    BUFFER_SIZE = 100000        # Replay buffer size (saved with checkpoint)
    TRAINING_EPOCHS = 10        # Epochs per training iteration

    # ==========================================================================
    # Paths
    # ==========================================================================
    CHECKPOINT_DIR = "checkpoints"
    DEFAULT_CHECKPOINT = "model_latest.pt"

    @classmethod
    def get_checkpoint_path(cls, filename: str = None) -> str:
        """Get full path to a checkpoint file."""
        if filename is None:
            filename = cls.DEFAULT_CHECKPOINT
        return os.path.join(cls.CHECKPOINT_DIR, filename)
