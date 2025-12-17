"""
Breakthrough Game Logic.

This module provides:
- 8x8 board with 16 pieces per player
- Simple action space (64 squares × 3 directions = 192 actions)
- 3-plane input features (my pieces, opponent pieces, ones)
- Canonical board representation (always from perspective of player to move)

Breakthrough Rules:
- Each player starts with 16 pieces on their first two rows
- Pieces move forward 1 square (straight or diagonal)
- Captures occur only on diagonal moves
- Win by reaching opponent's home row OR capturing all opponent pieces
- Draws are impossible
"""

import numpy as np
from typing import List, Optional, Tuple

from src.config import Config


# =============================================================================
# Constants
# =============================================================================

BOARD_SIZE = Config.BOARD_SIZE
WHITE = 1   # Player who moves first, starts on rows 0-1
BLACK = -1  # Player who moves second, starts on rows 6-7
EMPTY = 0

# Action encoding: 64 squares × 3 directions = 192 actions
# Direction 0: straight forward
# Direction 1: diagonal left-forward
# Direction 2: diagonal right-forward
NUM_ACTIONS = Config.NUM_ACTIONS

# Direction offsets (row_delta, col_delta) from WHITE's perspective
# WHITE moves towards higher rows, BLACK moves towards lower rows
DIRECTIONS = [
    (1, 0),   # Forward
    (1, -1),  # Diagonal left
    (1, 1),   # Diagonal right
]


def _encode_action(from_row: int, from_col: int, direction: int) -> int:
    """Encode a move as an action index."""
    square = from_row * BOARD_SIZE + from_col
    return square * 3 + direction


def _decode_action(action: int) -> Tuple[int, int, int]:
    """Decode an action index to (from_row, from_col, direction)."""
    square = action // 3
    direction = action % 3
    from_row = square // BOARD_SIZE
    from_col = square % BOARD_SIZE
    return from_row, from_col, direction


class BreakthroughGame:
    """
    Breakthrough game wrapper with simple state encoding.
    
    Features:
    - 192 action space (64 squares × 3 directions)
    - 3-plane input features
    - Canonical board representation
    """
    
    INPUT_PLANES = Config.INPUT_PLANES
    
    def __init__(self, board: Optional[np.ndarray] = None, turn: int = WHITE):
        """
        Initialize a new game.
        
        Args:
            board: Optional 8x8 numpy array. If None, creates starting position.
            turn: Which player moves next (WHITE=1 or BLACK=-1)
        """
        if board is not None:
            self.board = board.copy()
        else:
            self.board = self._initial_board()
        self.turn = turn
        self._winner: Optional[int] = None
    
    def _initial_board(self) -> np.ndarray:
        """Create the starting board position."""
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        # White pieces on rows 0-1
        board[0, :] = WHITE
        board[1, :] = WHITE
        # Black pieces on rows 6-7
        board[6, :] = BLACK
        board[7, :] = BLACK
        return board
    
    def clone(self) -> "BreakthroughGame":
        """Create a deep copy of the game state."""
        new_game = BreakthroughGame.__new__(BreakthroughGame)
        new_game.board = self.board.copy()
        new_game.turn = self.turn
        new_game._winner = self._winner
        return new_game
    
    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """
        Return list of legal moves as (from_row, from_col, to_row, to_col) tuples.
        """
        moves = []
        # Direction of movement depends on whose turn it is
        row_dir = 1 if self.turn == WHITE else -1
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row, col] != self.turn:
                    continue
                
                # Check each direction
                for dir_idx, (dr, dc) in enumerate(DIRECTIONS):
                    to_row = row + row_dir * dr
                    to_col = col + dc * row_dir  # Mirror column direction for black
                    
                    # Correct column direction handling
                    if self.turn == BLACK:
                        # For black, we negate row direction and mirror diagonals
                        to_row = row - dr
                        to_col = col - dc
                    else:
                        to_row = row + dr
                        to_col = col + dc
                    
                    if not self._is_valid_square(to_row, to_col):
                        continue
                    
                    target = self.board[to_row, to_col]
                    
                    if dir_idx == 0:
                        # Straight forward: must be empty
                        if target == EMPTY:
                            moves.append((row, col, to_row, to_col))
                    else:
                        # Diagonal: empty or enemy piece (capture)
                        if target != self.turn:  # Empty or opponent
                            moves.append((row, col, to_row, to_col))
        
        return moves
    
    def _is_valid_square(self, row: int, col: int) -> bool:
        """Check if coordinates are within the board."""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
    
    def step(self, move: Tuple[int, int, int, int]):
        """
        Apply a move to the board.
        
        Args:
            move: Tuple of (from_row, from_col, to_row, to_col)
        """
        from_row, from_col, to_row, to_col = move
        
        # Move the piece
        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = EMPTY
        
        # Check for win conditions
        self._check_winner(to_row)
        
        # Switch turn
        self.turn = -self.turn
    
    def _check_winner(self, to_row: int):
        """Check if the move resulted in a win."""
        # Win by reaching opponent's home row
        if self.turn == WHITE and to_row == BOARD_SIZE - 1:
            self._winner = WHITE
        elif self.turn == BLACK and to_row == 0:
            self._winner = BLACK
        
        # Win by capturing all opponent pieces
        if self._winner is None:
            opponent = -self.turn
            if not np.any(self.board == opponent):
                self._winner = self.turn
    
    def is_terminal(self) -> bool:
        """Check if the game is over."""
        if self._winner is not None:
            return True
        # Also check if current player has no legal moves (shouldn't happen normally)
        return len(self.get_legal_moves()) == 0
    
    def get_result(self) -> Tuple[float, float]:
        """
        Get game result as (win, loss) from WHITE's perspective.
        
        Returns:
            Tuple of (win_prob, loss_prob) for WHITE.
        """
        if not self.is_terminal():
            return (0.0, 0.0)
        
        if self._winner == WHITE:
            return (1.0, 0.0)
        elif self._winner == BLACK:
            return (0.0, 1.0)
        else:
            # No moves available - current player loses
            if self.turn == WHITE:
                return (0.0, 1.0)  # White loses
            else:
                return (1.0, 0.0)  # Black loses, so White wins
    
    def get_reward(self) -> float:
        """
        Get reward from WHITE's perspective.
        +1 for white win, -1 for black win, 0 for ongoing.
        """
        w, l = self.get_result()
        return w - l
    
    def encode_action(self, move: Tuple[int, int, int, int]) -> int:
        """
        Encode a move to an action index (0-191).
        
        The encoding is from the perspective of the current player.
        For black, the board is conceptually flipped.
        """
        from_row, from_col, to_row, to_col = move
        
        # Determine direction
        if self.turn == WHITE:
            dr = to_row - from_row
            dc = to_col - from_col
        else:
            # For black, flip perspective
            from_row = BOARD_SIZE - 1 - from_row
            from_col = BOARD_SIZE - 1 - from_col
            to_row_flip = BOARD_SIZE - 1 - to_row
            to_col_flip = BOARD_SIZE - 1 - to_col
            dr = to_row_flip - from_row
            dc = to_col_flip - from_col
        
        # Map delta to direction index
        if dc == 0:
            direction = 0  # Forward
        elif dc == -1:
            direction = 1  # Diagonal left
        else:
            direction = 2  # Diagonal right
        
        return _encode_action(from_row, from_col, direction)
    
    def decode_action(self, action: int) -> Tuple[int, int, int, int]:
        """
        Decode an action index to a move tuple.
        
        The action is from the network's perspective (always as if playing WHITE).
        For black, the move is transformed back to actual board coordinates.
        """
        from_row, from_col, direction = _decode_action(action)
        
        # Get direction deltas
        dr, dc = DIRECTIONS[direction]
        to_row = from_row + dr
        to_col = from_col + dc
        
        if self.turn == BLACK:
            # Flip back to actual board coordinates
            from_row = BOARD_SIZE - 1 - from_row
            from_col = BOARD_SIZE - 1 - from_col
            to_row = BOARD_SIZE - 1 - to_row
            to_col = BOARD_SIZE - 1 - to_col
        
        return (from_row, from_col, to_row, to_col)
    
    def get_encoded_state(self) -> np.ndarray:
        """
        Encode the current position as input planes.
        
        The encoding is always from the perspective of the player to move.
        For black, the board is flipped vertically and horizontally.
        
        Planes (3 total):
        - 0: Current player's pieces (1 where we have a piece)
        - 1: Opponent's pieces (1 where opponent has a piece)
        - 2: All ones (helps network detect board edges)
        """
        planes = np.zeros((self.INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        
        if self.turn == WHITE:
            # No flip needed
            planes[0] = (self.board == WHITE).astype(np.float32)
            planes[1] = (self.board == BLACK).astype(np.float32)
        else:
            # Flip board for black's perspective (180 degree rotation)
            flipped = np.flip(np.flip(self.board, 0), 1)
            planes[0] = (flipped == BLACK).astype(np.float32)
            planes[1] = (flipped == WHITE).astype(np.float32)
        
        # Plane 2: All ones
        planes[2] = 1.0
        
        return planes
    
    def get_legal_action_mask(self) -> np.ndarray:
        """Return a boolean mask of legal actions."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        for move in self.get_legal_moves():
            action = self.encode_action(move)
            mask[action] = True
        return mask
    
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {WHITE: '○', BLACK: '●', EMPTY: '.'}
        lines = []
        lines.append("  a b c d e f g h")
        for row in range(BOARD_SIZE - 1, -1, -1):
            row_str = f"{row + 1} "
            for col in range(BOARD_SIZE):
                row_str += symbols[self.board[row, col]] + " "
            lines.append(row_str)
        lines.append(f"Turn: {'White' if self.turn == WHITE else 'Black'}")
        return "\n".join(lines)


# Alias for compatibility
Game = BreakthroughGame


if __name__ == "__main__":
    # Quick test
    g = BreakthroughGame()
    print(f"Initial board:\n{g}")
    print(f"Legal moves: {len(g.get_legal_moves())}")
    
    # Test encoding
    state = g.get_encoded_state()
    print(f"State shape: {state.shape}")
    
    # Test action encoding/decoding
    moves = g.get_legal_moves()
    for move in moves[:5]:
        idx = g.encode_action(move)
        decoded = g.decode_action(idx)
        print(f"{move} -> {idx} -> {decoded}")
    
    # Play a few moves
    print("\nPlaying a few moves...")
    for i in range(4):
        moves = g.get_legal_moves()
        if not moves:
            break
        move = moves[0]
        print(f"Move {i+1}: {move}")
        g.step(move)
    
    print(f"\nAfter moves:\n{g}")
