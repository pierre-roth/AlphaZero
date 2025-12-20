
import pytest
import time
from src.baseline.state import BitboardState
from src.baseline.constants import WHITE, BLACK, SCORE_WIN
from src.baseline.search import Search

def test_initial_state():
    state = BitboardState()
    assert state.turn == WHITE
    # Initial moves: 16 pawns * 2 forward + 0 diagonal = 16? 
    # Or some diagonals possible? No, files adjacent.
    # A1(0) has B2(9) and empty ahead.
    # White on 0-15. Black on 48-63.
    # Row 1 (8-15) blocked by Row 2 (empty).
    # Forward moves from Row 1 are valid. Capture? No enemies.
    # Total moves: 16 forward?
    assert len(state.get_legal_moves()) > 0

def test_simple_move_generation():
    # Setup: White pawn on A2(8), Black on B3(17)
    # White to move.
    # Moves for A2: 
    # - Forward to A3(16): Valid if empty.
    # - Diag R to B3(17): Capture.
    # - Diag L: None (blocked by edge).
    
    state = BitboardState(white=1<<8, black=1<<17, turn=WHITE)
    moves = state.get_legal_moves()
    
    # Expected: (8->16) quiet, (8->17) calc
    move_pairs = sorted(moves)
    assert (8, 16) in move_pairs
    assert (8, 17) in move_pairs
    assert len(move_pairs) == 2

def test_win_detection():
    # White on Rank 7 (A7=48). Move to A8(56). Win.
    # Add a black pawn so it's not win-by-capture
    state = BitboardState(white=1<<48, black=1<<8, turn=WHITE)
    state.make_move(48, 56)
    
    # State check: White reached Rank 8?
    # Actually state.check_win() should return WHITE
    assert state.check_win() == WHITE
    
    # Search should see this as immediate win
    # Reset
    state = BitboardState(white=1<<48, black=1<<8, turn=WHITE)
    search = Search()
    # Depth 1 should capture the win
    best_move, score = search.search(state, time_ms=1000)
    print(f"DEBUG Test: best_move={best_move}, score={score}")
    assert best_move == (48, 56)
    assert score > 20000 # Near max score

def test_search_defence():
    # White about to win. Black must capture.
    # White at A7(48). Black at B7(49).
    # White moves A7->A8 win.
    # Black to move.
    # Black options: B7->A6 (no), B7->B6 (fwd), B7->A6(capture?)
    # Wait.
    # Setup: White A7(48). Black B8(57).
    # Black at B8 blocks A8? No A8 is 56. B8 is 57.
    # Setup:
    # White pawn at A7 (index 48).
    # Black pawn at B8 (index 57).
    # Black to move.
    # White threatens A7->A8 (56).
    # Black can capture A7 from B8?
    # B8(57) >> 9 = 48 (A7). Capture!
    # If Black does not capture, White wins next.
    # Search should find capture (57->48).
    
    state = BitboardState(white=1<<48, black=1<<57, turn=BLACK)
    search = Search()
    best_move, score = search.search(state, time_ms=500)
    
    # Must capture (57->48)
    assert best_move == (57, 48)

def test_perft_speed():
    # Just a quick check that we generate moves reasonably fast
    state = BitboardState()
    # 1000 random moves
    start = time.time()
    for _ in range(1000):
        moves = state.get_legal_moves()
        if not moves: break
        # pick first
        state.make_move(*moves[0])
    dur = time.time() - start
    print(f"1000 moves in {dur:.4f}s")
    assert dur < 1.0 # Should be very fast
