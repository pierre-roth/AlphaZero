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
    # Default model size (can be overridden by MODEL_SIZES)
    RESNET_BLOCKS = 6           # Number of residual blocks
    RESNET_FILTERS = 128        # Number of filters per conv layer
    SE_RATIO = 8                # Squeeze-Excitation reduction ratio
    
    # Available model sizes (blocks, filters)
    MODEL_SIZES = {
        'small': {'blocks': 5, 'filters': 64},
        'medium': {'blocks': 10, 'filters': 128},
        'large': {'blocks': 20, 'filters': 128},  # Deep > Wide for reasoning
    }
    DEFAULT_MODEL_SIZE = 'large'

    # ==========================================================================
    # MCTS Parameters
    # ==========================================================================
    MCTS_SIMULATIONS = 400      # Simulations per search (training) - reduced for 8x8 board
    MCTS_SIMULATIONS_INFERENCE = 800  # Simulations for web inference (stronger play)
    C_PUCT = 1.5                # Exploration constant
    # First Play Urgency: reduction from parent Q-value for unvisited nodes.
    # FPU = parent_Q - FPU_REDUCTION. Prevents "despair exploration" in losing positions.
    FPU_REDUCTION = 0.25
    DIRICHLET_ALPHA = 0.4        # Dirichlet noise alpha (higher = more exploration for small boards)
    DIRICHLET_EPSILON = 0.25    # Dirichlet noise weight
    TEMPERATURE_THRESHOLD = 15  # Moves before switching to deterministic

    # ==========================================================================
    # Training Parameters
    # ==========================================================================
    BATCH_SIZE = 512            # Larger batch for smoother gradients
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PARALLEL_GAMES = 64         # Games to run in parallel during self-play
    SELFPLAY_BATCHES = 4        # Number of self-play rounds before each training cycle
    BUFFER_SIZE = 100000        # Replay buffer size (saved with checkpoint)
    TRAINING_EPOCHS = 2         # Epochs per training iteration (reduced to prevent overfitting)

    # ==========================================================================
    # Paths
    # ==========================================================================
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL = "model_best.pt"        # Best model by arena ELO
    DATA_FILE = "training_data.npz"     # Training examples (append-only)
    ARENA_STATE = "arena_state.json"    # Arena ratings and match history

    @classmethod
    def get_checkpoint_path(cls, filename: str) -> str:
        """Get full path to a checkpoint file."""
        return os.path.join(cls.CHECKPOINT_DIR, filename)
