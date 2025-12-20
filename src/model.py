"""
Neural Network Architecture for AlphaZero Breakthrough.

This module implements an SE-ResNet (Squeeze-and-Excitation ResNet) for
the Breakthrough game.

Key features:
- 3 input channels (piece planes + auxiliary)
- 192 output actions (64 squares × 3 directions)
- WL value head (Win/Loss probabilities - no draws in Breakthrough)
- Squeeze-and-Excitation blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    
    Applies channel-wise attention by:
    1. Global average pooling (squeeze)
    2. FC -> ReLU -> FC -> Sigmoid (excitation)
    3. Scale original features by excitation weights
    """
    
    def __init__(self, channels: int, se_ratio: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // se_ratio)
        self.fc2 = nn.Linear(channels // se_ratio, channels * 2)
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Squeeze: global average pooling
        squeezed = self.pool(x).view(batch_size, -1)
        
        # Excitation: FC -> ReLU -> FC
        excited = F.relu(self.fc1(squeezed))
        excited = self.fc2(excited)
        
        # Split into weight and bias
        w, b = excited.split(self.channels, dim=1)
        w = torch.sigmoid(w).view(batch_size, self.channels, 1, 1)
        b = b.view(batch_size, self.channels, 1, 1)
        
        # Scale and shift
        return x * w + b


class SEResidualBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation.
    
    Structure:
    - Conv 3x3 -> BN -> ReLU
    - Conv 3x3 -> BN
    - SE Block
    - Skip connection + ReLU
    """
    
    def __init__(self, channels: int, se_ratio: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_ratio)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE
        out = self.se(out)
        
        # Skip connection
        out = out + residual
        out = F.relu(out)
        
        return out


class AlphaZeroNet(nn.Module):
    """
    AlphaZero neural network for Breakthrough.
    
    Args:
        num_blocks: Number of residual blocks (default: 6)
        num_filters: Number of filters/channels (default: 128)
        num_input_planes: Input channels (default: 3)
        num_actions: Output actions (default: 192)
        se_ratio: Squeeze-Excitation ratio (default: 8)
    """
    
    def __init__(
        self,
        num_blocks: int = Config.RESNET_BLOCKS,
        num_filters: int = Config.RESNET_FILTERS,
        num_input_planes: int = Config.INPUT_PLANES,
        num_actions: int = Config.OUTPUT_ACTIONS,
        se_ratio: int = Config.SE_RATIO,
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.num_input_planes = num_input_planes
        self.num_actions = num_actions
        
        # Input block
        self.input_conv = nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.residual_tower = nn.ModuleList([
            SEResidualBlock(num_filters, se_ratio) for _ in range(num_blocks)
        ])
        
        # Policy head
        # Conv -> BN -> ReLU -> Flatten -> Linear
        self.policy_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(num_filters)
        self.policy_fc = nn.Linear(num_filters * 8 * 8, num_actions)
        
        # Value head (WL - Win/Loss, no draws possible in Breakthrough)
        # Conv 1x1 -> BN -> ReLU -> Flatten -> Linear -> ReLU -> Linear(2)
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 2)  # WL output (Win/Loss)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, 8, 8)
            
        Returns:
            policy_logits: Shape (batch, 192) - move probabilities (apply softmax)
            wl: Shape (batch, 2) - Win/Loss probabilities (apply softmax)
        """
        # Input block
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)
        
        # Residual tower
        for block in self.residual_tower:
            out = block(out)
        
        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = F.relu(value)
        wl = self.value_fc2(value)  # Raw logits, apply softmax externally
        
        return policy_logits, wl
    
    def get_value(self, wl_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert WL logits to a single value in [-1, 1].
        
        Args:
            wl_logits: Shape (batch, 2) - Win/Loss logits
            
        Returns:
            value: Shape (batch, 1) - Expected value in [-1, 1]
        """
        wl_probs = F.softmax(wl_logits, dim=1)
        # Value = P(win) - P(loss)
        value = wl_probs[:, 0] - wl_probs[:, 1]
        return value.unsqueeze(1)


# Backwards compatibility alias
LC0Network = AlphaZeroNet





if __name__ == "__main__":
    # Test the model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = AlphaZeroNet(num_blocks=6, num_filters=128).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 8, 8).to(device)
    policy, wdl = model(dummy_input)
    
    print(f"Policy shape: {policy.shape}")  # (2, 192)
    print(f"WL shape: {wdl.shape}")  # (2, 2) - Win/Loss only, no draws in Breakthrough
    
    # Test value conversion
    value = model.get_value(wdl)
    print(f"Value shape: {value.shape}")  # (2, 1)
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    
    print("Model test passed!")
