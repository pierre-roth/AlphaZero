from .constants import (
    WHITE, BLACK, BOARD_SIZE,
    RANK_7, RANK_2,
    FILE_A, FILE_B, FILE_G, FILE_H,
    SCORE_WIN, SCORE_LOSS, SCORE_PAWN
)
from .state import BitboardState

def evaluate(state: BitboardState) -> int:
    """
    Evaluate the position from White's perspective.
    Returns score relative to White (positive means White is better).
    """
    # 1. Immediate Win Checks (Redundant with search but safe)
    winner = state.check_win()
    if winner == WHITE:
        return SCORE_WIN
    if winner == BLACK:
        return SCORE_LOSS

    score = 0
    
    # --- Material ---
    # +100 * pawn_count
    w_count = bin(state.white).count('1')
    b_count = bin(state.black).count('1')
    score += SCORE_PAWN * (w_count - b_count)
    
    # --- Advancement (Rank Progress) ---
    # White pawn on rank r (1..8): +12 * (r-1)
    # Black pawn on rank r: +12 * (8-r)
    # Ranks are 0-indexed in logic (0..7)
    # White: Rank 0 -> +0, Rank 1 -> +12, ..., Rank 6 -> +72
    # Black: Rank 7 -> +0, Rank 6 -> +12, ..., Rank 1 -> +72
    
    # Iterate White pawns
    w = state.white
    while w:
        lsb = w & -w
        idx = lsb.bit_length() - 1
        rank = idx // 8
        score += 12 * rank
        w ^= lsb
        
    # Iterate Black pawns
    b = state.black
    while b:
        lsb = b & -b
        idx = lsb.bit_length() - 1
        rank = idx // 8
        score -= 12 * (7 - rank)
        b ^= lsb
        
    # --- Centralization ---
    # +4 for each pawn on files C-F (Files 2,3,4,5)
    # Mask for C-F
    FILE_C = 0x0101010101010101 << 2
    FILE_F = 0x0101010101010101 << 5
    # Range C(2) to F(5)
    # Actually explicit mask: C|D|E|F
    MASK_CENTER = (0x0101010101010101 << 2) | \
                  (0x0101010101010101 << 3) | \
                  (0x0101010101010101 << 4) | \
                  (0x0101010101010101 << 5)

    w_center = bin(state.white & MASK_CENTER).count('1')
    b_center = bin(state.black & MASK_CENTER).count('1')
    score += 4 * (w_center - b_center)
    
    # --- Mobility ---
    # +4 * (legalMovesWhite - legalMovesBlack)
    # This is expensive to compute fully.
    # We can approximate or just generate them.
    # Since we need this fast, let's skip expensive full generation if we want speed.
    # BUT the spec says: "+4 * (legalMovesWhite - legalMovesBlack)".
    # For a baseline, correctness matches spec. Let's do it properly but optimize later if needed.
    # However, generating moves for the side NOT to move requires swapping turn temporarily or implementing logic.
    # The `get_legal_moves` function depends on `self.turn`.
    
    # Save turn
    original_turn = state.turn
    
    # Calc White Mobility
    state.turn = WHITE
    w_moves = len(state.get_legal_moves())
    
    # Calc Black Mobility
    state.turn = BLACK
    b_moves = len(state.get_legal_moves())
    
    # Restore turn
    state.turn = original_turn
    
    score += 4 * (w_moves - b_moves)
    
    # --- Protected Pawns ---
    # +10 per protected pawn (diagonal support behind)
    # White protected by (rank-1, col+/-1)
    # "Diagonal support behind" means if A2 is there, B3 is supported? YES.
    # So B3 is supported if A2 or C2 exists.
    # Implementation:
    # Supported positions for White = (White << 7 | White << 9) & White
    # Note: Mask wrapping?
    # W << 7 (Up-Left support). Dest on File H invalid? No, source on File A invalid.
    # Correct: Support FROM W.
    # If W at A1(0), supports B2(9) via +9.
    # If W at B1(1), supports A2(8) via +7 and C2(10) via +9.
    # So White Supported Set = ( (W << 7) & ~FILE_H ) | ( (W << 9) & ~FILE_A )
    # Overlap with W to count protected pawns.
    
    w_attacks = ((state.white << 7) & ~FILE_H) | ((state.white << 9) & ~FILE_A)
    w_protected = w_attacks & state.white
    score += 10 * bin(w_protected).count('1')
    
    # Black protected by (rank+1, col+/-1) => (Black >> 7 | Black >> 9)
    # B >> 7 (Down-Right). Source H8(63) -> A7(56). Wrap! Must mask dest?
    # Source H, dest A.
    # Correct: (B >> 7) & ~FILE_A
    # Correct: (B >> 9) & ~FILE_H
    
    b_attacks = ((state.black >> 7) & ~FILE_A) | ((state.black >> 9) & ~FILE_H)
    b_protected = b_attacks & state.black
    score -= 10 * bin(b_protected).count('1')
    
    # --- Hanging Pawns ---
    # If pawn is capturable and NOT defended: -25
    # If capturable BUT defended: -10
    
    # White hanging: in `b_attacks`
    w_in_danger = state.white & b_attacks
    # Split into defended and undefended
    w_defended_danger = w_in_danger & w_protected
    w_undefended_danger = w_in_danger & ~w_protected
    
    score -= 10 * bin(w_defended_danger).count('1')
    score -= 25 * bin(w_undefended_danger).count('1')
    
    # Black hanging: in `w_attacks`
    b_in_danger = state.black & w_attacks
    b_defended_danger = b_in_danger & b_protected
    b_undefended_danger = b_in_danger & ~b_protected
    
    score += 10 * bin(b_defended_danger).count('1')
    score += 25 * bin(b_undefended_danger).count('1')
    
    # --- Near-Promotion Threat ---
    # White on Rank 6 (index 48-55) => threaten Rank 7 (Win)
    # The logic in spec says "White rank 7". Wait.
    # "White rank 7 / Black rank 2"
    # User likely means 1-based ranks 1..8.
    # So Rank 7 is rows 6 in 0-indexed. (0..7). Yes.
    # White on Row 6.
    
    # Base bonus: +180
    w_near_prom = state.white & RANK_7
    w_near_count = bin(w_near_prom).count('1')
    score += 180 * w_near_count
    
    b_near_prom = state.black & RANK_2
    b_near_count = bin(b_near_prom).count('1')
    score -= 180 * b_near_count
    
    # If it has at least one empty/winning step to promotion next ply: +260
    # For White at Row 6:
    # Steps are: Forward (Row 7), DiagL (Row 7), DiagR (Row 7).
    # If ANY is legal (Empty or Capture), it's a threat.
    # We masked w_near_prom already.
    # Check moves from these pieces.
    # Since they are on Row 6, any move goes to Row 7.
    # Any move to Row 7 is a WIN immediately next turn.
    # So if `get_legal_moves` returns any move starting from these squares?
    # Or simpler: Can it move forward? can it capture?
    #   Forward: (sq+8) is empty?
    #   DiagL: (sq+7) is empty/enemy?
    #   DiagR: (sq+9) is empty/enemy?
    
    # Let's verify individually to count correctly (since "can stack" implies per pawn?)
    # "total can stack" -> yes, per pawn.
    
    # We can do this bitwise.
    # White Threatening Move Sources:
    # 1. Forwardable: (w_near_prom << 8) & empty -> Sources are (Result >> 8)
    occ = state.white | state.black
    empty = ~occ
    
    # Sources that can move forward
    w_can_fwd = ( (w_near_prom << 8) & empty ) >> 8
    
    # Sources that can move DiagL (Capture or Quiet)
    # Quiet: (w << 7) & empty. Capture: (w << 7) & black.
    # Combined: (w << 7) & ~white.
    # Dest must be on Rank 7 (implicit if source is Rank 6) and NOT File H.
    w_can_dl = ( ( (w_near_prom << 7) & ~FILE_H ) & ~state.white ) >> 7
    
    # Sources DiagR
    w_can_dr = ( ( (w_near_prom << 9) & ~FILE_A ) & ~state.white ) >> 9
    
    # Union of sources that have AT LEAST one move
    w_threats = w_can_fwd | w_can_dl | w_can_dr
    score += 260 * bin(w_threats).count('1')
    
    # Black Threats (Row 1)
    # Forward: -8. DiagL: -9. DiagR: -7.
    b_can_fwd = ( (b_near_prom >> 8) & empty ) << 8
    
    # DiagL (>> 9), mask ~FILE_H dest
    b_can_dl = ( ( (b_near_prom >> 9) & ~FILE_H ) & ~state.black ) << 9
    
    # DiagR (>> 7), mask ~FILE_A dest
    b_can_dr = ( ( (b_near_prom >> 7) & ~FILE_A ) & ~state.black ) << 7
    
    b_threats = b_can_fwd | b_can_dl | b_can_dr
    score -= 260 * bin(b_threats).count('1')
    
    # --- Promotion Race Bonus ---
    # White: + (8-rank) ... wait.
    # "White: + (8 - rank) is distance; bonus + (70 - 10*distance) clipped at 0"
    # Rank is 1-based (1..8).
    # Distance to Rank 8.
    # If on Rank 7 (Row 6): Dist = 1. Bonus = 70 - 10 = 60.
    # If on Rank 6 (Row 5): Dist = 2. Bonus = 70 - 20 = 50.
    # If on Rank 1 (Row 0): Dist = 7. Bonus = 70 - 70 = 0.
    # Formula: Bonus = 70 - 10 * (8 - (row+1)) = 70 - 10*(7-row).
    # For each pawn.
    
    w = state.white
    while w:
        lsb = w & -w
        idx = lsb.bit_length() - 1
        row = idx // 8
        dist = 7 - row
        bonus = max(0, 70 - 10 * dist)
        score += bonus
        w ^= lsb
        
    b = state.black
    while b:
        lsb = b & -b
        idx = lsb.bit_length() - 1
        row = idx // 8
        dist = row  # Distance to Row 0 is just row index
        bonus = max(0, 70 - 10 * dist)
        score -= bonus
        b ^= lsb

    return score
