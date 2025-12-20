"""
Web Interface for AlphaZero Breakthrough.

Flask backend that serves a Breakthrough game interface and handles bot moves.
"""


from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
import os
import glob
from typing import Optional, Tuple, Literal
import numpy as np

from src.model import AlphaZeroNet
from src.mcts import MCTS
from src.game import BreakthroughGame, WHITE, BLACK, BOARD_SIZE, EMPTY
from src.config import Config
from src.baseline.state import BitboardState
from src.baseline.search import Search
from src.baseline.constants import WHITE as BASELINE_WHITE, BLACK as BASELINE_BLACK

app = Flask(__name__, static_folder='../web', static_url_path='')
CORS(app)

# Global game state and bot
device = None
model = None
mcts = None
current_game = None
current_model_name = None
baseline_search = Search(time_limit_ms=1000)  # 1 second think time

# Game Modes
MODE_ALPHAZERO = 'alphazero'
MODE_BASELINE = 'baseline'
MODE_AI_VS_BASELINE = 'ai_vs_baseline'

current_mode = MODE_ALPHAZERO
white_player_type = 'human'
black_player_type = 'alphazero'

def get_available_models():
    """List available model checkpoints."""
    models = []
    checkpoint_dir = Config.CHECKPOINT_DIR
    
    if os.path.exists(checkpoint_dir):
        # Find all .pt files
        for f in glob.glob(os.path.join(checkpoint_dir, "*.pt")):
            name = os.path.basename(f)
            size = os.path.getsize(f)
            models.append({
                'name': name,
                'path': f,
                'size_mb': round(size / (1024 * 1024), 2),
            })
    
    # Sort by name
    models.sort(key=lambda x: x['name'])
    
    return models


def load_model(model_path: str) -> tuple[bool, str]:
    """
    Load a model from checkpoint with dynamic architecture detection.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    global model, mcts, current_model_name, device
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Default architecture from config
    num_blocks = Config.RESNET_BLOCKS
    num_filters = Config.RESNET_FILTERS
    
    if not os.path.exists(model_path):
        print(f"No checkpoint found at {model_path}, using random weights")
        model = AlphaZeroNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
        current_model_name = "random"
        model.eval()
        mcts = MCTS(model, num_simulations=Config.MCTS_SIMULATIONS_INFERENCE, device=str(device))
        return True, "Using random weights (no checkpoint found)"
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Try to get architecture from checkpoint
        if 'config' in checkpoint:
            num_blocks = checkpoint['config'].get('num_blocks', num_blocks)
            num_filters = checkpoint['config'].get('num_filters', num_filters)
        
        model = AlphaZeroNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        current_model_name = os.path.basename(model_path)
        model.eval()
        mcts = MCTS(model, num_simulations=Config.MCTS_SIMULATIONS_INFERENCE, device=str(device))
        print(f"Loaded model from {model_path} ({num_blocks} blocks, {num_filters} filters)")
        return True, f"Loaded {current_model_name}"
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        print(error_msg)
        return False, error_msg


def init_bot():
    """
    Initialize with the best available model.
    
    Priority:
    1. model_best.pt (highest ELO from arena)
    2. Latest iteration_N.pt checkpoint
    3. Random weights (fallback)
    """
    import glob
    import re
    
    checkpoint_dir = Config.CHECKPOINT_DIR
    
    # Default architecture
    num_blocks = Config.RESNET_BLOCKS
    num_filters = Config.RESNET_FILTERS
    
    # Try best model first
    best_path = os.path.join(checkpoint_dir, Config.BEST_MODEL)
    if os.path.exists(best_path):
        print(f"Loading best model ({Config.BEST_MODEL})...")
        load_model(best_path)
        return
    
    # Fall back to latest iteration checkpoint
    pattern = os.path.join(checkpoint_dir, "iteration_*.pt")
    files = glob.glob(pattern)
    
    if files:
        iterations = []
        for f in files:
            match = re.search(r'iteration_(\d+)\.pt$', f)
            if match:
                iterations.append((int(match.group(1)), f))
        
        if iterations:
            iterations.sort(reverse=True)
            latest_path = iterations[0][1]
            print(f"No best model found, loading latest iteration checkpoint...")
            load_model(latest_path)
            return
    
    # Fall back to random weights
    print(f"No checkpoints found, using random weights...")
    global model, mcts, current_model_name, device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AlphaZeroNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    current_model_name = "random"
    model.eval()
    mcts = MCTS(model, num_simulations=Config.MCTS_SIMULATIONS_INFERENCE, device=str(device))





# =============================================================================
# Translation Layer (AlphaZero <-> Baseline)
# =============================================================================

def game_to_baseline(game: BreakthroughGame) -> BitboardState:
    """Convert AlphaZero BreakthroughGame to Baseline BitboardState."""
    white_bb = 0
    black_bb = 0
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = game.board[r, c]
            if piece == WHITE:
                # Map (r, c) to bit index (r*8 + c)
                idx = r * 8 + c
                white_bb |= (1 << idx)
            elif piece == BLACK:
                idx = r * 8 + c
                black_bb |= (1 << idx)
                
    turn = BASELINE_WHITE if game.turn == WHITE else BASELINE_BLACK
    
    return BitboardState(white=white_bb, black=black_bb, turn=turn)

def baseline_move_to_game(move: Tuple[int, int], turn: int) -> Tuple[int, int, int, int]:
    """Convert Baseline (from_sq, to_sq) to AlphaZero (r1, c1, r2, c2)."""
    from_sq, to_sq = move
    
    r1 = from_sq // 8
    c1 = from_sq % 8
    r2 = to_sq // 8
    c2 = to_sq % 8
    
    return (r1, c1, r2, c2)


def make_baseline_move_logic() -> dict:
    """
    Execute a move using the Baseline engine.
    Returns the response dictionary similar to bot_move.
    """
    global current_game, baseline_search
    
    # Convert state
    bb_state = game_to_baseline(current_game)
    
    # Search
    # Use 2000ms typically
    best_move, score = baseline_search.search(bb_state, time_ms=2000)
    
    if best_move is None:
        # Should not happen unless game over
        return {'error': 'Baseline returned no move'}
        
    # Apply move to main game
    game_move = baseline_move_to_game(best_move, current_game.turn)
    current_game.step(game_move)
    
    response = board_to_json(current_game)
    response['bot_move'] = list(game_move)
    # Hide evaluation for baseline
    response['evaluation'] = None 
    response['game_over'] = current_game.is_terminal()
    response['result'] = get_result() if current_game.is_terminal() else None
    response['legal_moves'] = get_legal_moves_json()
    
    return response


def board_to_json(game: BreakthroughGame) -> dict:
    """Convert board state to JSON-serializable format."""
    board_list = []
    for row in range(BOARD_SIZE):
        row_list = []
        for col in range(BOARD_SIZE):
            cell = int(game.board[row, col])
            row_list.append(cell)
        board_list.append(row_list)
    
    return {
        'board': board_list,
        'turn': 'white' if game.turn == WHITE else 'black',
    }


@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models."""
    models = get_available_models()
    return jsonify({
        'models': models,
        'current': current_model_name
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Return game configuration for the frontend."""
    return jsonify({
        'board_size': Config.BOARD_SIZE,
        'num_actions': Config.NUM_ACTIONS,
    })


@app.route('/api/models/select', methods=['POST'])
def select_model():
    """Select a model to use."""
    data = request.get_json() or {}
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({'error': 'No model specified'}), 400
    
    model_path = os.path.join(Config.CHECKPOINT_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model not found: {model_name}'}), 404
    
    success, message = load_model(model_path)
    
    if not success:
        return jsonify({'error': message}), 500
    
    return jsonify({
        'success': True,
        'current': current_model_name,
        'message': message
    })


@app.route('/api/new', methods=['POST'])
def new_game():
    """Start a new game with specified player types."""
    global current_game, white_player_type, black_player_type
    
    data = request.get_json() or {}
    white_type = data.get('white_type', 'human') # 'human', 'alphazero', 'baseline'
    black_type = data.get('black_type', 'alphazero')
    
    # Set global player types
    white_player_type = white_type
    black_player_type = black_type

    # Load AlphaZero model if needed
    if 'alphazero' in [white_player_type, black_player_type]:
        # Only re-init if really necessary? For now just rely on current loaded model
        # or we could force a reload if we want to ensure "best" is used?
        # But 'init_bot' is expensive. Let's assume the user selects model via proper endpoint or we rely on what's loaded.
        # But the old code called init_bot_for_size(model_size).
        # We can just ensure a model is loaded.
        if model is None:
            init_bot()
    
    current_game = BreakthroughGame()
    
    response = board_to_json(current_game)
    response['white_type'] = white_player_type
    response['black_type'] = black_player_type
    response['game_over'] = False
    response['model'] = current_model_name
    response['legal_moves'] = get_legal_moves_json()
    
    # If White is a bot, make move immediately
    if white_player_type != 'human':
        bot_response = resolve_bot_move()
        response.update(bot_response)
        
    return jsonify(response)


def get_legal_moves_json():
    """Get legal moves as list of [from_row, from_col, to_row, to_col]."""
    if current_game is None:
        return []
    return [list(move) for move in current_game.get_legal_moves()]


def resolve_bot_move():
    """Decide which bot to call based on current turn."""
    turn = current_game.turn
    player_type = white_player_type if turn == WHITE else black_player_type
    
    if player_type == 'alphazero':
        return make_alphazero_move()
    elif player_type == 'baseline':
        return make_baseline_move_logic()
    else:
        return {'error': 'It is human turn'}


@app.route('/api/move', methods=['POST'])
def player_move():
    """Handle player move and respond with bot move."""
    global current_game
    
    if current_game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    data = request.get_json()
    move_data = data.get('move')
    
    if not move_data or len(move_data) != 4:
        return jsonify({'error': 'Invalid move format'}), 400
    
    from_row, from_col, to_row, to_col = move_data
    move = (from_row, from_col, to_row, to_col)
    
    # Validate move
    legal_moves = current_game.get_legal_moves()
    if move not in legal_moves:
        return jsonify({'error': 'Illegal move'}), 400
    
    current_game.step(move)
    
    # Check if game over after player move
    if current_game.is_terminal():
        response = board_to_json(current_game)
        response['game_over'] = True
        response['result'] = get_result()
        response['legal_moves'] = []
        response['moved_player'] = 'white' if current_game.turn == BLACK else 'black' # The player who JUST moved
        return jsonify(response)
    
    # Identify who just moved (the turn has already flipped in game.step)
    # If turn is now BLACK, then WHITE just moved.
    just_moved = 'white' if current_game.turn == BLACK else 'black'
    
    response = board_to_json(current_game)
    response['moved_player'] = just_moved
    
    # Check if next player is a bot
    next_turn = current_game.turn
    next_player_type = white_player_type if next_turn == WHITE else black_player_type
    
    if next_player_type != 'human':
        # Bot responds
        bot_response = resolve_bot_move()
        response.update(bot_response)
    else:
        # Next is human, just return state
        response['legal_moves'] = get_legal_moves_json()
        
    return jsonify(response)


@app.route('/api/bot_move', methods=['POST'])
def bot_move_endpoint():
    """Trigger a bot move for the current player."""
    global current_game
    
    if current_game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    if current_game.is_terminal():
        return jsonify({'error': 'Game already finished'}), 400
    
    bot_response = resolve_bot_move()
    return jsonify(bot_response)


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current game state."""
    global current_game
    
    if current_game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    response = board_to_json(current_game)
    response['game_over'] = current_game.is_terminal()
    response['result'] = get_result() if current_game.is_terminal() else None
    response['legal_moves'] = get_legal_moves_json()
    response['model'] = current_model_name
    
    return jsonify(response)


def make_alphazero_move():
    """Make a move using MCTS and return the new state."""
    global current_game, mcts
    
    # Run MCTS search
    root = mcts.search(current_game, add_noise=False)
    
    # Get best action
    best_action = -1
    best_visits = -1
    for action, child in root.children.items():
        if child.visit_count > best_visits:
            best_visits = child.visit_count
            best_action = action
    
    # Get evaluation (convert to absolute: positive = White winning)
    # root.value() is from current player's perspective, so negate for Black
    eval_score = root.value() if root.children else 0
    if current_game.turn == BLACK:
        eval_score = -eval_score
    
    # Make move
    move = current_game.decode_action(best_action)
    current_game.step(move)
    
    response = board_to_json(current_game)
    response['bot_move'] = list(move)
    response['evaluation'] = float(eval_score)
    response['game_over'] = current_game.is_terminal()
    response['result'] = get_result() if current_game.is_terminal() else None
    response['legal_moves'] = get_legal_moves_json()
    
    return response


def get_result():
    """Get the game result string."""
    if not current_game.is_terminal():
        return None
    
    w, l = current_game.get_result()
    if w == 1.0:
        return 'White wins!'
    elif l == 1.0:
        return 'Black wins!'
    else:
        return None  # Game not finished


# Initialize bot when module loads
init_bot()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5051, debug=True)
