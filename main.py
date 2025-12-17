"""
AlphaZero Breakthrough Training and Inference.

Usage:
    python main.py train          - Start training with parallel self-play
    python main.py web            - Start web server
"""

import sys
import torch

# Disable torch.compile to avoid slow sympy imports (~150s delay)
torch._dynamo.config.suppress_errors = True
import torch._dynamo
torch._dynamo.disable()

from src.model import AlphaZeroNet
from src.parallel_trainer import ParallelTrainer


from src.config import Config

def train():
    """
    Train with parallel self-play.
    
    The training loop is restartable - it automatically resumes from the
    latest iteration checkpoint and reloads training data from disk.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network
    model = AlphaZeroNet(num_blocks=Config.RESNET_BLOCKS, num_filters=Config.RESNET_FILTERS).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    trainer = ParallelTrainer(
        model,
        device=str(device),
        num_parallel_games=Config.PARALLEL_GAMES,
        num_simulations=Config.MCTS_SIMULATIONS
    )
    
    # Find and load latest iteration checkpoint
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
        print(f"Iteration {iteration}")
        print(f"{'='*50}")
        
        # Parallel Self Play
        new_examples = trainer.execute_parallel_episodes(num_games=Config.PARALLEL_GAMES)
        
        # Add to in-memory replay buffer (handles size limit internally)
        trainer.add_examples(new_examples)
        
        # Train
        print(f"\nTraining on {len(trainer.examples)} examples...")
        trainer.learn(epochs=Config.TRAINING_EPOCHS, batch_size=Config.BATCH_SIZE)
        
        # Save training data to disk (appends all new examples)
        trainer.append_training_data(new_examples)
        
        # Save iteration checkpoint (model-only, for restart)
        trainer.save_iteration_checkpoint(iteration)


def arena():
    """Run the arena for model evaluation."""
    from src.arena import run_arena
    run_arena()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "train":
            train()
        elif cmd == "web":
            from src.web import app
            print("Starting web server at http://localhost:5051")
            app.run(host="0.0.0.0", port=5051, debug=False)
        elif cmd == "arena":
            arena()
        else:
            print("Usage: python main.py [train|web|arena]")
    else:
        print("Usage: python main.py [train|web|arena]")
