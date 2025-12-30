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
Start the self-play training loop:
```bash
python main.py train
```
- **Resumable**: Automatically continues from the latest checkpoint.
- **Parallel**: Runs 100+ games simultaneously for efficiency.
- **Persistence**: Saves training data to `checkpoints/training_data.npz`.

### 3. Arena (Evaluation)
Run the automated evaluation system:
```bash
python main.py arena
```
- Continuously scans for new checkpoints.
- Estimates ELO ratings by playing matches against the best previous models.
- Tracks a leaderboard in `checkpoints/arena_state.json`.

### 4. Web Interface
Play against the AI in your browser:
```bash
python main.py web
```
- Open **http://localhost:5051**
- **Features**:
  - **Human vs Bot**: Test your skills against AlphaZero or the Baseline engine.
  - **Human vs Human**: Play locally with move validation and history.
  - **Bot vs Bot**: Watch models play against each other.


## Network Architecture

- **Input:** 3 planes (8×8): my pieces, opponent pieces, ones
- **Tower:** 20 ResNet blocks with 128 filters
- **Policy head:** 192 outputs (64 squares × 3 directions)
- **Value head:** WL (Win/Loss) probabilities

## Running Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All settings are centralized in `src/config.py`.


Important note: Currently FPU is 0 for all unvisited nodes. This was used during early training. Going forward, ParentQ - const should be considered instead. 

