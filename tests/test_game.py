"""
Tests for game.py - Breakthrough Game Logic.
"""

import pytest
import numpy as np

from src.game import (
    BreakthroughGame, WHITE, BLACK, EMPTY, BOARD_SIZE,
    NUM_ACTIONS, _encode_action, _decode_action
)


class TestActionEncoding:
    """Tests for action encoding/decoding."""
    
    def test_action_count(self):
        """Verify action space size."""
        assert NUM_ACTIONS == 192  # 64 squares × 3 directions
    
    def test_encode_decode_roundtrip(self):
        """Encode then decode should give same values."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                for direction in range(3):
                    action = _encode_action(row, col, direction)
                    decoded_row, decoded_col, decoded_dir = _decode_action(action)
                    assert decoded_row == row
                    assert decoded_col == col
                    assert decoded_dir == direction
    
    def test_action_range(self):
        """All actions should be in valid range."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                for direction in range(3):
                    action = _encode_action(row, col, direction)
                    assert 0 <= action < NUM_ACTIONS


class TestBreakthroughGameSetup:
    """Tests for initial game setup."""
    
    def test_initial_board(self):
        """Initial board should have correct piece placement."""
        game = BreakthroughGame()
        
        # White pieces on rows 0-1
        for col in range(BOARD_SIZE):
            assert game.board[0, col] == WHITE
            assert game.board[1, col] == WHITE
        
        # Black pieces on rows 6-7
        for col in range(BOARD_SIZE):
            assert game.board[6, col] == BLACK
            assert game.board[7, col] == BLACK
        
        # Middle rows empty
        for row in range(2, 6):
            for col in range(BOARD_SIZE):
                assert game.board[row, col] == EMPTY
    
    def test_initial_turn(self):
        """White should move first."""
        game = BreakthroughGame()
        assert game.turn == WHITE
    
    def test_piece_count(self):
        """Each player should have 16 pieces."""
        game = BreakthroughGame()
        assert np.sum(game.board == WHITE) == 16
        assert np.sum(game.board == BLACK) == 16


class TestBreakthroughGameMoves:
    """Tests for move generation."""
    
    def test_initial_legal_moves_count(self):
        """White should have legal moves from starting position."""
        game = BreakthroughGame()
        moves = game.get_legal_moves()
        # Each piece in row 1 can move forward and potentially diagonal
        # Row 1 has 8 pieces, each can move forward (8 moves)
        # Plus diagonal moves from edge pieces
        assert len(moves) > 0
    
    def test_forward_move(self):
        """Piece should be able to move straight forward."""
        game = BreakthroughGame()
        moves = game.get_legal_moves()
        
        # Check that forward moves exist
        forward_moves = [m for m in moves if m[1] == m[3] and m[2] == m[0] + 1]
        assert len(forward_moves) > 0
    
    def test_diagonal_move(self):
        """Piece should be able to move diagonally forward."""
        game = BreakthroughGame()
        moves = game.get_legal_moves()
        
        # Check that diagonal moves exist
        diagonal_moves = [m for m in moves if m[1] != m[3]]
        assert len(diagonal_moves) > 0
    
    def test_no_backward_moves(self):
        """Pieces should not be able to move backward."""
        game = BreakthroughGame()
        moves = game.get_legal_moves()
        
        for move in moves:
            from_row, _, to_row, _ = move
            # White moves toward higher rows
            assert to_row > from_row
    
    def test_capture_only_diagonal(self):
        """Captures should only happen on diagonal moves."""
        # Create a position where capture is possible
        game = BreakthroughGame()
        # Place a black piece in front of a white piece diagonally
        game.board[2, 1] = BLACK  # Enemy piece
        game.board[1, 0] = WHITE  # Our piece
        
        moves = game.get_legal_moves()
        
        # The piece at (1, 0) should be able to capture at (2, 1)
        capture_move = (1, 0, 2, 1)
        assert capture_move in moves


class TestBreakthroughGameStep:
    """Tests for making moves."""
    
    def test_step_moves_piece(self):
        """Making a move should update the board."""
        game = BreakthroughGame()
        move = game.get_legal_moves()[0]
        
        from_row, from_col, to_row, to_col = move
        game.step(move)
        
        assert game.board[from_row, from_col] == EMPTY
        assert game.board[to_row, to_col] == WHITE
    
    def test_step_switches_turn(self):
        """Making a move should switch turns."""
        game = BreakthroughGame()
        assert game.turn == WHITE
        
        move = game.get_legal_moves()[0]
        game.step(move)
        
        assert game.turn == BLACK
    
    def test_capture_removes_piece(self):
        """Capturing should remove the opponent's piece."""
        game = BreakthroughGame()
        # Set up a capture scenario
        game.board[2, 1] = BLACK
        game.board[1, 0] = WHITE
        
        initial_black_count = np.sum(game.board == BLACK)
        
        capture_move = (1, 0, 2, 1)
        game.step(capture_move)
        
        assert np.sum(game.board == BLACK) == initial_black_count - 1


class TestBreakthroughGameWinConditions:
    """Tests for win conditions."""
    
    def test_win_by_reaching_home_row(self):
        """White should win by reaching row 7."""
        game = BreakthroughGame()
        # Clear the board except for one white piece about to win
        game.board[:, :] = EMPTY
        game.board[6, 4] = WHITE  # One step from winning
        game.turn = WHITE
        
        move = (6, 4, 7, 4)
        game.step(move)
        
        assert game.is_terminal()
        w, l = game.get_result()
        assert w == 1.0  # White wins
    
    def test_win_by_capturing_all(self):
        """Player should win by capturing all opponent pieces."""
        game = BreakthroughGame()
        # Clear board, leave one piece of each
        game.board[:, :] = EMPTY
        game.board[3, 3] = WHITE
        game.board[4, 4] = BLACK  # Last black piece
        game.turn = WHITE
        
        # Capture the last black piece
        move = (3, 3, 4, 4)
        game.step(move)
        
        assert game.is_terminal()
        w, l = game.get_result()
        assert w == 1.0  # White wins
    
    def test_not_terminal_at_start(self):
        """Game should not be terminal at start."""
        game = BreakthroughGame()
        assert not game.is_terminal()
    
    def test_black_win(self):
        """Black should win by reaching row 0."""
        game = BreakthroughGame()
        game.board[:, :] = EMPTY
        game.board[1, 4] = BLACK
        game.turn = BLACK
        
        move = (1, 4, 0, 4)
        game.step(move)
        
        assert game.is_terminal()
        w, l = game.get_result()
        assert l == 1.0  # Black wins = White loses


class TestBreakthroughGameClone:
    """Tests for cloning."""
    
    def test_clone_is_independent(self):
        """Clone should be independent copy."""
        game1 = BreakthroughGame()
        game2 = game1.clone()
        
        # Modify clone
        move = game2.get_legal_moves()[0]
        game2.step(move)
        
        # Original should be unchanged
        assert game1.turn == WHITE
        assert game2.turn == BLACK


class TestBreakthroughGameEncoding:
    """Tests for state encoding."""
    
    def test_state_shape(self):
        """State should be (4, 8, 8)."""
        game = BreakthroughGame()
        state = game.get_encoded_state()
        
        assert state.shape == (3, 8, 8)
        assert state.dtype == np.float32
    
    def test_my_pieces_plane(self):
        """Plane 0 should show current player's pieces."""
        game = BreakthroughGame()
        state = game.get_encoded_state()
        
        # White's turn, so plane 0 should have white pieces
        assert state[0, 0, :].sum() == 8  # Row 0 has 8 white pieces
        assert state[0, 1, :].sum() == 8  # Row 1 has 8 white pieces
    
    def test_opponent_pieces_plane(self):
        """Plane 1 should show opponent's pieces."""
        game = BreakthroughGame()
        state = game.get_encoded_state()
        
        # White's turn, so plane 1 should have black pieces
        assert state[1, 6, :].sum() == 8  # Row 6 has 8 black pieces
        assert state[1, 7, :].sum() == 8  # Row 7 has 8 black pieces
    
    def test_ones_plane(self):
        """Plane 2 should be all ones."""
        game = BreakthroughGame()
        state = game.get_encoded_state()
        
        assert np.all(state[2] == 1.0)
    
    def test_perspective_flip_for_black(self):
        """Board should be flipped for black's perspective."""
        game = BreakthroughGame()
        game.step(game.get_legal_moves()[0])  # Now black's turn
        
        state = game.get_encoded_state()
        
        # From black's perspective (flipped), black pieces should be in rows 0-1
        # This is because the board is 180-degree rotated
        assert state[0].sum() > 0  # Black's pieces (now "my pieces")


class TestActionEncodeDecode:
    """Tests for move encoding/decoding integration."""
    
    def test_encode_decode_roundtrip_white(self):
        """Encoding then decoding should give same move for white."""
        game = BreakthroughGame()
        
        for move in game.get_legal_moves():
            action = game.encode_action(move)
            decoded = game.decode_action(action)
            assert move == decoded, f"Move {move} became {decoded}"
    
    def test_encode_decode_roundtrip_black(self):
        """Encoding then decoding should work for black too."""
        game = BreakthroughGame()
        game.step(game.get_legal_moves()[0])  # Now black's turn
        
        for move in game.get_legal_moves():
            action = game.encode_action(move)
            decoded = game.decode_action(action)
            assert move == decoded, f"Move {move} became {decoded}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
