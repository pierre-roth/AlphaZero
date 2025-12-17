# AlphaZero Breakthrough

An AlphaZero-style neural network for the abstract strategy game **Breakthrough**.

## What is Breakthrough?

Breakthrough is a two-player game played on an 8×8 board. Each player starts with 16 pieces filling their first two rows.

**Rules:**
- Pieces move one square forward (straight or diagonally)
- Captures are made only on diagonal moves
- First player to reach the opponent's home row **wins**
- Capturing all opponent pieces also wins
- No draws are possible!

## Project Structure

- `src/game.py` — Breakthrough game logic, state encoding, action mapping
- `src/model.py` — Neural network (SE-ResNet with policy and value heads)
- `src/mcts.py` — Monte Carlo Tree Search with PUCT and Dirichlet noise
- `src/parallel_trainer.py` — Parallel self-play and training loop
- `src/web.py` — Flask web server for playing against the bot
- `src/config.py` — Centralized hyperparameters
- `web/` — Frontend (HTML/CSS/JS)

## How to Run

### 1. Setup
```bash
cd /path/to/AlphaZero
source venv/bin/activate
pip install torch numpy flask flask-cors tqdm
```

### 2. Training
```bash
python main.py train
```
- Runs parallel self-play games
- Trains the neural network on collected examples
- Saves checkpoints to `checkpoints/`

### 3. Play in Browser
```bash
python main.py web
```
Open http://localhost:5051 and play against the bot!

## Network Architecture

- **Input:** 3 planes (8×8): my pieces, opponent pieces, ones
- **Tower:** 6 SE-ResNet blocks with 128 filters
- **Policy head:** 192 outputs (64 squares × 3 directions)
- **Value head:** WL (Win/Loss) probabilities

## Running Tests

```bash
python -m pytest tests/ -v
```

## Configuration

Edit `src/config.py` to tune:
- Network size (`RESNET_BLOCKS`, `RESNET_FILTERS`)
- MCTS parameters (`MCTS_SIMULATIONS`, `C_PUCT`)
- Training settings (`BATCH_SIZE`, `LEARNING_RATE`, `PARALLEL_GAMES`)
