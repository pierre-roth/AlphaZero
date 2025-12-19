"""
Tests for parallel_trainer.py - AlphaZero Training logic.
"""

import os
import pytest
import torch
import numpy as np
from src.parallel_trainer import ParallelTrainer
from src.model import AlphaZeroNet
from src.config import Config

class TestParallelTrainer:
    """Tests for ParallelTrainer class."""

    @pytest.fixture
    def device(self):
        return "cpu"

    @pytest.fixture
    def model(self):
        return AlphaZeroNet(num_blocks=2, num_filters=32)

    @pytest.fixture
    def trainer(self, model, device):
        return ParallelTrainer(model, device=device, num_parallel_games=1, num_simulations=1, model_size="small")

    def test_lr_scheduler_t_max_override(self, trainer):
        """
        Verify that ParallelTrainer respects the new T_max even when loading 
        a state_dict that might have a different default or saved value.
        """
        # 1. Check initial T_max is what we expect from Config
        assert trainer.scheduler.T_max == Config.LR_SCHEDULER_T_MAX
        
        # 2. Simulate a checkpoint with T_max = 1000
        # We manually create a state dict for the scheduler
        state_dict = trainer.scheduler.state_dict()
        state_dict['T_max'] = 1000
        
        # 3. Save this dummy checkpoint
        checkpoint_filename = "iteration_1_small.pt"
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_filename)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        
        torch.save({
            'state_dict': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': state_dict,
            'iteration': 1,
            'model_size': "small"
        }, checkpoint_path)
        
        try:
            # 4. Load the checkpoint using the trainer's method
            # This should trigger our new logic: load_state_dict THEN override T_max
            success = trainer.load_iteration_checkpoint(1)
            assert success is True
            
            # 5. Verify T_max is overridden to Config.LR_SCHEDULER_T_MAX (200)
            assert trainer.scheduler.T_max == Config.LR_SCHEDULER_T_MAX
            assert trainer.scheduler.T_max == 200
            
        finally:
            # Cleanup
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
