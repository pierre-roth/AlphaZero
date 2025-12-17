"""
AlphaZero Breakthrough Training and Inference.

Usage:
    python main.py train [--size small|medium|large]  - Start training
    python main.py web                                 - Start web server
    python main.py arena                               - Run model evaluation
"""

import sys
import argparse
import torch

# Disable torch.compile to avoid slow sympy imports (~150s delay)
torch._dynamo.config.suppress_errors = True
import torch._dynamo
torch._dynamo.disable()

from src.model import AlphaZeroNet
from src.parallel_trainer import ParallelTrainer


from src.config import Config


def get_model_config(size: str) -> tuple:
    """Get blocks and filters for a model size."""
    if size not in Config.MODEL_SIZES:
        raise ValueError(f"Unknown model size: {size}. Use: {list(Config.MODEL_SIZES.keys())}")
    cfg = Config.MODEL_SIZES[size]
    return cfg['blocks'], cfg['filters']


def train(model_size: str = Config.DEFAULT_MODEL_SIZE):
    """
    Train with parallel self-play.
    
    The training loop is restartable - it automatically resumes from the
    latest iteration checkpoint and reloads training data from disk.
    
    Args:
        model_size: Model size to train ('small', 'medium', 'large')
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get model configuration
    num_blocks, num_filters = get_model_config(model_size)
    print(f"Model size: {model_size} ({num_blocks} blocks, {num_filters} filters)")
    
    # Create network
    model = AlphaZeroNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    trainer = ParallelTrainer(
        model,
        device=str(device),
        num_parallel_games=Config.PARALLEL_GAMES,
        num_simulations=Config.MCTS_SIMULATIONS,
        model_size=model_size  # Pass size for checkpoint naming
    )
    
    # Find and load latest iteration checkpoint for this model size
    start_iteration = trainer.get_latest_iteration()
    
    if start_iteration > 0:
        print(f"\nResuming from iteration {start_iteration}...")
        trainer.load_iteration_checkpoint(start_iteration)
        trainer.load_training_data()  # Loads last 100k examples
    else:
        print("\nStarting fresh training run...")
    
    print("Starting training loop with parallel self-play...")
    iteration = start_iteration
    
    while True:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"Iteration {iteration} ({model_size})")
        print(f"{'='*50}")
        
        # Collect multiple batches of self-play data before training
        all_new_examples = []
        for batch_idx in range(Config.SELFPLAY_BATCHES):
            print(f"\nSelf-play batch {batch_idx + 1}/{Config.SELFPLAY_BATCHES}")
            new_examples = trainer.execute_parallel_episodes(num_games=Config.PARALLEL_GAMES)
            all_new_examples.extend(new_examples)
            trainer.add_examples(new_examples)
        
        print(f"\nCollected {len(all_new_examples)} new examples from {Config.SELFPLAY_BATCHES} batches")
        
        # Train
        print(f"\nTraining on {len(trainer.examples)} examples...")
        trainer.learn(epochs=Config.TRAINING_EPOCHS, batch_size=Config.BATCH_SIZE)
        
        # Save training data to disk (appends all new examples)
        trainer.append_training_data(all_new_examples)
        
        # Save iteration checkpoint (model-only, for restart)
        trainer.save_iteration_checkpoint(iteration)


def arena(cross_size: bool = False):
    """Run the arena for model evaluation."""
    if cross_size:
        from src.arena import run_cross_size_arena
        run_cross_size_arena()
    else:
        from src.arena import run_arena
        run_arena()


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Breakthrough')
    parser.add_argument('command', choices=['train', 'web', 'arena'], 
                        help='Command to run')
    parser.add_argument('--size', choices=['small', 'medium', 'large'],
                        default=Config.DEFAULT_MODEL_SIZE,
                        help='Model size for training (default: large)')
    parser.add_argument('--cross-size', action='store_true',
                        help='Arena mode: pit best models of different sizes against each other')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(model_size=args.size)
    elif args.command == 'web':
        from src.web import app
        print("Starting web server at http://localhost:5051")
        app.run(host="0.0.0.0", port=5051, debug=False)
    elif args.command == 'arena':
        arena(cross_size=args.cross_size)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python main.py [train|web|arena] [--size small|medium|large] [--cross-size]")
    else:
        main()

