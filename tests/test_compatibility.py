
import numpy as np
import pytest
from src.game import BreakthroughGame, WHITE as G_WHITE, BLACK as G_BLACK
from src.baseline.state import BitboardState
from src.baseline.constants import WHITE, BLACK

def game_to_bitboard(game):
    w = 0
    b = 0
    for r in range(8):
        for c in range(8):
            if game.board[r, c] == G_WHITE:
                w |= (1 << (r * 8 + c))
            elif game.board[r, c] == G_BLACK:
                b |= (1 << (r * 8 + c))
    return BitboardState(white=w, black=b, turn=WHITE if game.turn == G_WHITE else BLACK)

def bitboard_to_game(state):
    board = np.zeros((8, 8), dtype=np.int8)
    for r in range(8):
        for f in range(8):
            mask = 1 << (r * 8 + f)
            if state.white & mask:
                board[r, f] = G_WHITE
            elif state.black & mask:
                board[r, f] = G_BLACK
    return BreakthroughGame(board=board, turn=G_WHITE if state.turn == WHITE else G_BLACK)

def test_initial_state():
    game = BreakthroughGame()
    state = BitboardState()
    
    # Compare boards
    game_state = game_to_bitboard(game)
    assert game_state.white == state.white
    assert game_state.black == state.black
    assert game_state.turn == state.turn

def test_moves_initial():
    game = BreakthroughGame()
    state = BitboardState()
    
    g_moves = set((r1*8+c1, r2*8+c2) for (r1,c1,r2,c2) in game.get_legal_moves())
    b_moves = set(state.get_legal_moves())
    
    assert g_moves == b_moves

def test_random_positions():
    import random
    rng = random.Random(42)
    
    for i in range(20):
        # Generate random position
        w = 0
        b = 0
        for sq in range(64):
            # Row 0 and 7 should be empty for a non-terminated start
            row = sq // 8
            if row == 0 or row == 7: continue
            
            p = rng.random()
            if p < 0.1: w |= (1 << sq)
            elif p < 0.2: b |= (1 << sq)
            
        for turn in [WHITE, BLACK]:
            state = BitboardState(white=w, black=b, turn=turn)
            game = bitboard_to_game(state)
            
            g_moves = set((r1*8+c1, r2*8+c2) for (r1,c1,r2,c2) in game.get_legal_moves())
            b_moves = set(state.get_legal_moves())
            
            if g_moves != b_moves:
                print(f"Mismatch at iteration {i}, turn {'WHITE' if turn==WHITE else 'BLACK'}")
                print(f"Board:\n{state}")
                print(f"G moves: {sorted(list(g_moves))}")
                print(f"B moves: {sorted(list(b_moves))}")
                assert g_moves == b_moves

def test_terminal_agreement():
    # Test reach back rank
    # White move from A7(48) to A8(56)
    state = BitboardState(white=1<<48, black=1<<8, turn=WHITE)
    game = bitboard_to_game(state)
    
    state.make_move(48, 56)
    game.step((6, 0, 7, 0)) # A7 to A8
    
    assert state.is_terminal() == game.is_terminal()
    assert state.check_win() == (WHITE if game._winner == G_WHITE else BLACK if game._winner == G_BLACK else None)
    assert state.check_win() == WHITE

    # Test capture all
    # White capture last black piece
    # White A2(8) captures Black B3(17)
    state = BitboardState(white=1<<8, black=1<<17, turn=WHITE)
    game = bitboard_to_game(state)
    
    state.make_move(8, 17)
    game.step((1, 0, 2, 1))
    
    assert state.is_terminal() == game.is_terminal()
    assert state.check_win() == (WHITE if game._winner == G_WHITE else BLACK if game._winner == G_BLACK else None)
    assert state.check_win() == WHITE
