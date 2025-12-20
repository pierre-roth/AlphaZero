"""
Constants for the Baseline Breakthrough Engine.
"""

# Board dimensions
BOARD_SIZE = 8
NUM_SQUARES = 64

# Players
WHITE = 1
BLACK = -1

# Bitboard Masks
FILE_A = 0x0101010101010101
FILE_B = FILE_A << 1
FILE_C = FILE_A << 2
FILE_D = FILE_A << 3
FILE_E = FILE_A << 4
FILE_F = FILE_A << 5
FILE_G = FILE_A << 6
FILE_H = FILE_A << 7

# Ranks
RANK_1 = 0x00000000000000FF
RANK_2 = RANK_1 << 8
RANK_3 = RANK_1 << 16
RANK_4 = RANK_1 << 24
RANK_5 = RANK_1 << 32
RANK_6 = RANK_1 << 40
RANK_7 = RANK_1 << 48
RANK_8 = RANK_1 << 56

# Special zones in bitboards
CENTER_FILES = FILE_C | FILE_D | FILE_E | FILE_F

# Scores (centipawns)
SCORE_PAWN = 100
SCORE_WIN = 30000
SCORE_LOSS = -30000
