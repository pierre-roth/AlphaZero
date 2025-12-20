import random
from typing import List, Tuple, Optional, Any

from .constants import (
    WHITE, BLACK, BOARD_SIZE, NUM_SQUARES,
    FILE_A, FILE_H, RANK_1, RANK_8,
    SCORE_WIN, SCORE_LOSS
)

class BitboardState:
    """
    Breakthrough state representation using bitboards.
    
    Attributes:
        white (int): Bitboard for white pawns.
        black (int): Bitboard for black pawns.
        turn (int): Current player (WHITE or BLACK).
        zobrist_key (int): Current Zobrist hash key.
    """
    
    # Precomputed Zobrist random numbers
    # [color_index][square_index] where color_index 0=White, 1=Black
    _ZOBRIST_PIECES: List[List[int]] = []
    _ZOBRIST_SIDE: int = 0
    
    @classmethod
    def _init_zobrist(cls):
        """Initialize Zobrist hashing constants."""
        if cls._ZOBRIST_PIECES:
            return
            
        rng = random.Random(42)  # Fixed seed for reproducibility
        # 0: White pawn, 1: Black pawn
        cls._ZOBRIST_PIECES = [[rng.getrandbits(64) for _ in range(64)] for _ in range(2)]
        cls._ZOBRIST_SIDE = rng.getrandbits(64)

    def __init__(self, white: int = 0xFFFF, black: int = 0xFFFF000000000000, turn: int = WHITE):
        """
        Initialize the state. Default is the standard starting position.
        
        Args:
            white: Bitboard of white pawns (default: rows 0-1)
            black: Bitboard of black pawns (default: rows 6-7)
            turn: Side to move (default: WHITE)
        """
        self._init_zobrist()
        
        self.white = white
        self.black = black
        self.turn = turn
        
        self.zobrist_key = self._compute_zobrist()

    def _compute_zobrist(self) -> int:
        """Compute the Zobrist key from scratch."""
        h = 0
        
        # White pieces
        w = self.white
        while w:
            lsb = w & -w
            sq = lsb.bit_length() - 1
            h ^= self._ZOBRIST_PIECES[0][sq]
            w ^= lsb
            
        # Black pieces
        b = self.black
        while b:
            lsb = b & -b
            sq = lsb.bit_length() - 1
            h ^= self._ZOBRIST_PIECES[1][sq]
            b ^= lsb
            
        # Side to move
        if self.turn == BLACK:
            h ^= self._ZOBRIST_SIDE
            
        return h

    def clone(self) -> 'BitboardState':
        """Create a deep copy of the state."""
        # Fast manual copy
        new_state = BitboardState.__new__(BitboardState)
        new_state.white = self.white
        new_state.black = self.black
        new_state.turn = self.turn
        new_state.zobrist_key = self.zobrist_key
        return new_state

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Generate all pseudo-legal moves.
        Returns a list of (from_sq, to_sq) tuples.
        """
        moves = []
        
        occ = self.white | self.black
        empty = ~occ
        
        if self.turn == WHITE:
            # Shift amounts for white:
            # Forward: +8 (<< 8)
            # Diag Left: +7 (<< 7)
            # Diag Right: +9 (<< 9)
            
            # --- Forward (Quiet) ---
            # destinations = (white << 8) & empty
            dst_f = (self.white << 8) & empty & 0xFFFFFFFFFFFFFFFF
            
            # Iterate through destination bits
            w = dst_f
            while w:
                to_bit = w & -w
                to_sq = to_bit.bit_length() - 1
                from_sq = to_sq - 8
                moves.append((from_sq, to_sq))
                w ^= to_bit
            
            # --- Diag Left (Capture or Quiet) ---
            # destinations = (white << 7) & ~FILE_H & ~white (cannot capture own)
            # Note regarding "Diag Left": From lower index to higher, +7 moves A1->Index7 which is different.
            # Visualizing board:
            # 56 57 .. 63
            # ..
            # 0  1  .. 7
            # White moves "North" (increasing index).
            # +7 is typically "Up-Left" (from H1(7) to G2(14)? No, 7+7=14. correct)
            # BUT wrapping: H1(7) << 7 -> 14 (G2). Wait. 7 is H1. 7+7=14 is G2.
            # A1(0) << 7 -> 7 (H1). This wraps around? NO.
            # A1 is file 0. H1 is file 7.
            # Rank 0: 0..7
            # Rank 1: 8..15
            # Up-Left from index sq: sq + 7.
            # Valid if sq is NOT in File A (index 0, 8, etc)?
            # File A is 0, 8, 16.
            # If sq=8 (A2), 8+7=15 (H2). That is a wrap-around from left to right!
            # So Up-Left (+7) must NOT be from File A.
            # Let's re-verify specific instructions:
            # "Diagonal-left (quiet or capture): dstDL = (W << 7) & ~FILE_H & ~W"
            # This logic says: Result shouldn't be on File H.
            # Let's trace: A1(0) << 7 -> 7 (H1). Wait. A1 to H1 is definitely wrong.
            # White moves NORTH (increasing rank). A1 is rank 0.
            # A1(0) to B2(9) is +9.
            # A1(0) to (null) is -1/7? No.
            # B1(1) to A2(8) is +7.
            # So +7 is "Up-Left" (File B -> File A).
            # The destination of a +7 move CANNOT be on File H (because that would mean wrap around from File A, e.g. A1(0)->H1 wrapped?).
            # Wait. A1(0) simply has no left diagonal.
            # B1(1) + 7 = 8 (A2). Correct.
            # H1(7) + 7 = 14 (G2). Correct for H->G.
            # A2(8) + 7 = 15 (H2). Wait. A2(8) is on Left Edge. It should NOT go to H2.
            # So: Moves generating via +7 MUST mask out moves ORIGINATING from File A? Or ending on File H?
            # If we shift ALL W << 7, then:
            # A2(8) << 7 -> 15 (H2). This is invalid.
            # So we must NOT allow destinations on File H?
            # 15 is File H. Correct.
            # So dstDL = (W << 7) & ~FILE_H. This seems correct.
            
            # Diagonal Left (+7)
            # Destination cannot be on File H (wrap around from A)
            dst_dl = (self.white << 7) & ~FILE_H & ~self.white & 0xFFFFFFFFFFFFFFFF
            
            w = dst_dl
            while w:
                to_bit = w & -w
                to_sq = to_bit.bit_length() - 1
                from_sq = to_sq - 7
                
                # Check target content
                # "Forward-diagonal if target is empty ... or opponent pawn"
                # If target is own pawn, it's masked out by ~self.white already.
                moves.append((from_sq, to_sq))
                w ^= to_bit

            # --- Diag Right (+9) ---
            # Destination cannot be on File A (wrap around from H)
            # H1(7) + 9 = 16 (A3). Wait.
            # H1(7) is right edge. 7+9 = 16 (A2).
            # H1 should not go to A2.
            # So dstDR = (W << 9) & ~FILE_A & ~W
            dst_dr = (self.white << 9) & ~FILE_A & ~self.white & 0xFFFFFFFFFFFFFFFF
            
            w = dst_dr
            while w:
                to_bit = w & -w
                to_sq = to_bit.bit_length() - 1
                from_sq = to_sq - 9
                moves.append((from_sq, to_sq))
                w ^= to_bit

        else: # BLACK's turn
            # Shift amounts for black (moving SOUTH, decreasing index):
            # Forward: -8 (>> 8)
            # Diag Right (Black's perspective right, i.e. South-West?):
            # White's perspective: South-East (towards H1) or South-West (towards A1).
            # Let's stick to board coordinates.
            # Black moves Rank 7 -> Rank 6 -> ...
            # South: -8.
            
            # Diag vs File Wraps:
            # B2(9) - 9 = 0 (A1). This is South-West (File B -> File A).
            # B2(9) - 7 = 2 (C1). This is South-East (File B -> File C).
            
            # "Diagonal-left (from Black’s perspective)"
            # Black stands on Rank 7 facing Rank 0.
            # "Left" for Black is towards File H (White's Right).
            # So Black's "Left" is South-East (-7)?
            # Let's check user spec: "dstDL = (B >> 9) & ~FILE_H & ~B"
            # (B >> 9) is -9.
            # -9 is B2(9)->A1(0). File B -> File A.
            # From White's view, this is South-West.
            # From Black's view (facing South), File A is on their RIGHT hand?
            # Standard chess: White at bottom. A is left. H is right.
            # Black at top. A is Black's Right. H is Black's Left.
            # So "Black's Left" is towards File H.
            # Movement towards File H implies index change?
            # G2(14) -> H1(7) is -7.
            # So -7 is Black's Left (South-East).
            # -9 is Black's Right (South-West).
            
            # USER SPEC: "Diagonal-left (from Black’s perspective): dstDL = (B >> 9) & ~FILE_H & ~B"
            # Wait. If >>9 is B2->A1, destination is A1 (File A).
            # Why mask ~FILE_H?
            # Maybe I am flipping "Left/Right" in my head vs User spec.
            # Let's trust the BITWISE LOGIC from the user spec primarily, but verify safety.
            
            # User says: dstDL = (B >> 9) & ~FILE_H
            # A8(56) >> 9 = 47 (H6).
            # A(File 0) -> H(File 7). This is a wrap!
            # So if shifting by 9 (South-West / Right-Down), A wraps to H.
            # So we MUST mask out FILE_H from the DESTINATION to avoid A->H wrap.
            # Yes, (B >> 9) & ~FILE_H is correct safety mask for -9 shift.
            
            # User says: "Diagonal-right: dstDR = (B >> 7) & ~FILE_A & ~B"
            # H8(63) >> 7 = 56 (A7).
            # H(File 7) -> A(File 0). This is a wrap!
            # So -7 (South-East / Left-Down) wraps H->A.
            # So we MUST mask out FILE_A from DESTINATION.
            # Yes, match is correct.
            
            # --- Forward ---
            dst_f = (self.black >> 8) & empty & 0xFFFFFFFFFFFFFFFF
            
            w = dst_f
            while w:
                to_bit = w & -w
                to_sq = to_bit.bit_length() - 1
                from_sq = to_sq + 8
                moves.append((from_sq, to_sq))
                w ^= to_bit
                
            # --- Diag Left (User spec: >> 9) ---
            dst_dl = (self.black >> 9) & ~FILE_H & ~self.black & 0xFFFFFFFFFFFFFFFF
            
            w = dst_dl
            while w:
                to_bit = w & -w
                to_sq = to_bit.bit_length() - 1
                from_sq = to_sq + 9
                moves.append((from_sq, to_sq))
                w ^= to_bit
                
            # --- Diag Right (User spec: >> 7) ---
            dst_dr = (self.black >> 7) & ~FILE_A & ~self.black & 0xFFFFFFFFFFFFFFFF
            
            w = dst_dr
            while w:
                to_bit = w & -w
                to_sq = to_bit.bit_length() - 1
                from_sq = to_sq + 7
                moves.append((from_sq, to_sq))
                w ^= to_bit
                
        return moves

    def is_terminal(self) -> bool:
        """Check if game is over."""
        # Win by reaching back rank
        if self.white & RANK_8:
            return True
        if self.black & RANK_1:
            return True
        
        # Win by capturing all pieces
        if self.white == 0 or self.black == 0:
            return True
            
        return False
        
    def check_win(self) -> Optional[int]:
        """
        Return winner color if game is over, else None.
        Checks:
        1. Current player reaching back rank (Immediate win)
        2. Opponent having 0 pieces (Immediate win)
        """
        # Note: We check connection to user rule:
        # "White wins if to \in RANK_8"
        # "Black wins if to \in RANK_1"
        
        if self.white & RANK_8:
            return WHITE
        if self.black & RANK_1:
            return BLACK
            
        if self.black == 0:
            return WHITE
        if self.white == 0:
            return BLACK
            
        return None

    def make_move(self, from_sq: int, to_sq: int):
        """
        Execute a move on the board.
        Does NOT check legality. Updates hash and turn.
        """
        from_mask = 1 << from_sq
        to_mask = 1 << to_sq
        move_mask = from_mask | to_mask
        
        # Determine piece type and capture
        if self.turn == WHITE:
            # Move white pawn
            self.white ^= move_mask
            # Update hash for specific pieces
            self.zobrist_key ^= self._ZOBRIST_PIECES[0][from_sq]
            self.zobrist_key ^= self._ZOBRIST_PIECES[0][to_sq]
            
            # Check capture
            if self.black & to_mask:
                self.black ^= to_mask
                # Remove black pawn from hash
                self.zobrist_key ^= self._ZOBRIST_PIECES[1][to_sq]
                
        else:
            # Move black pawn
            self.black ^= move_mask
            # Update hash for specific pieces
            self.zobrist_key ^= self._ZOBRIST_PIECES[1][from_sq]
            self.zobrist_key ^= self._ZOBRIST_PIECES[1][to_sq]
            
            # Check capture
            if self.white & to_mask:
                self.white ^= to_mask
                # Remove white pawn from hash
                self.zobrist_key ^= self._ZOBRIST_PIECES[0][to_sq]

        # Switch turn and hash side
        self.turn = -self.turn
        self.zobrist_key ^= self._ZOBRIST_SIDE

    def __str__(self) -> str:
        """String representation."""
        lines = []
        lines.append("  A B C D E F G H")
        for r in range(7, -1, -1):
            line_content = []
            for f in range(8):
                idx = r * 8 + f
                mask = 1 << idx
                if self.white & mask:
                    line_content.append("W")
                elif self.black & mask:
                    line_content.append("B")
                else:
                    line_content.append(".")
            lines.append(f"{r+1} {' '.join(line_content)}")
        
        turn_str = "White" if self.turn == WHITE else "Black"
        lines.append(f"Turn: {turn_str}")
        lines.append(f"Key: {self.zobrist_key:016X}")
        return "\n".join(lines)
