/**
 * AlphaZero Breakthrough - Frontend JavaScript
 * 
 * Handles board rendering, move input, and communication with the Flask backend.
 */

// Game configuration (loaded from API)
let BOARD_SIZE = 8;  // Default, will be updated from /api/config
const WHITE = 1;
const BLACK = -1;
const EMPTY = 0;

// Game state
let currentBoard = null;
let playerColor = 'white';
let selectedSquare = null;
let legalMoves = [];
let moveHistory = [];
let isThinking = false;
let moveCount = 0;
let isAiVsAi = false;
let stopAiFlag = false;

// DOM Elements
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const movesEl = document.getElementById('moves');
const evalFillEl = document.getElementById('evalFill');
const evalLabelEl = document.getElementById('evalLabel');

// Initialize
async function init() {
    // Fetch config from backend
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        BOARD_SIZE = config.board_size;
    } catch (error) {
        console.warn('Could not fetch config, using defaults:', error);
    }

    // Setup event listeners
    const startBtn = document.getElementById('startGame');
    const stopBtn = document.getElementById('stopGame');

    if (startBtn) startBtn.addEventListener('click', startGame);
    if (stopBtn) stopBtn.addEventListener('click', () => { stopAiFlag = true; });

    // Initial render
    initializeEmptyBoard();
}

init();

/**
 * Start a new game
 */
async function startGame() {
    // Get configuration
    const whiteType = document.getElementById('whitePlayer').value;
    const blackType = document.getElementById('blackPlayer').value;
    const modelSize = document.getElementById('modelSize').value || 'medium';

    // Determine if we need auto-play loop (if NO humans involved)
    // Actually, even if Human vs Bot, we don't need a loop, the bot moves on response.
    // The LOOP is only needed if BOTH are bots.
    isAiVsAi = (whiteType !== 'human' && blackType !== 'human');

    stopAiFlag = false;
    moveHistory = [];
    selectedSquare = null;
    moveCount = 0;

    // Determine player color for UI perspective
    // If Human vs Bot, perspective is Human.
    // If Human vs Human, perspective is White (or toggle?). Let's default White.
    // If Bot vs Bot, perspective is White.
    if (whiteType === 'human' && blackType !== 'human') {
        playerColor = 'white';
    } else if (whiteType !== 'human' && blackType === 'human') {
        playerColor = 'black';
    } else {
        playerColor = 'white'; // Default perspective
    }

    // Toggle buttons
    document.getElementById('startGame').style.display = isAiVsAi ? 'none' : 'block';
    document.getElementById('stopGame').style.display = isAiVsAi ? 'block' : 'none';

    // Disable selectors during game? Optional enhancement.

    setStatus(`Starting game (${whiteType} vs ${blackType})...`);

    try {
        const response = await fetch('/api/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                white_type: whiteType,
                black_type: blackType,
                size: modelSize
            })
        });

        const data = await response.json();
        currentBoard = data.board;
        legalMoves = data.legal_moves || [];

        // Store types for local logic
        // We might need to know who is human locally to allowing clicking
        window.gameConfig = { white: whiteType, black: blackType };

        if (data.bot_move) {
            addMove(data.bot_move, 'white'); // White moved
            updateEval(data.evaluation);
        }

        renderBoard();
        updateStatus(data);

        if (isAiVsAi) {
            gameLoop();
        }

    } catch (error) {
        console.error('Error starting game:', error);
        setStatus('Error starting game.');
        resetControls();
    }

    // Update eval bar orientation
    const evalBar = document.querySelector('.eval-bar');
    if (playerColor === 'black') {
        evalBar.classList.add('flipped');
    } else {
        evalBar.classList.remove('flipped');
    }
}

function resetControls() {
    document.getElementById('startGame').style.display = 'block';
    document.getElementById('stopGame').style.display = 'none';
    isAiVsAi = false;
}

/**
 * Render the game board
 */
function renderBoard() {
    if (!currentBoard) return;

    boardEl.innerHTML = '';

    const isFlipped = playerColor === 'black';

    for (let displayRow = 0; displayRow < BOARD_SIZE; displayRow++) {
        for (let displayCol = 0; displayCol < BOARD_SIZE; displayCol++) {
            // Map display position to actual board position
            const row = isFlipped ? displayRow : (BOARD_SIZE - 1 - displayRow);
            const col = isFlipped ? (BOARD_SIZE - 1 - displayCol) : displayCol;

            const piece = currentBoard[row][col];
            createSquare(row, col, displayRow, displayCol, piece);
        }
    }
}

/**
 * Create a single square element
 */
function createSquare(row, col, displayRow, displayCol, piece) {
    const square = document.createElement('div');
    const isLight = (displayRow + displayCol) % 2 === 0;

    square.className = `square ${isLight ? 'light' : 'dark'}`;
    square.dataset.row = row;
    square.dataset.col = col;

    // Add piece if present
    if (piece !== EMPTY) {
        const pieceEl = document.createElement('div');
        pieceEl.className = `piece ${piece === WHITE ? 'white-piece' : 'black-piece'}`;
        square.appendChild(pieceEl);
    }

    // Highlight selected square
    if (selectedSquare && selectedSquare.row === row && selectedSquare.col === col) {
        square.classList.add('selected');
    }

    // Show legal move indicators
    if (selectedSquare) {
        const isLegalMove = legalMoves.some(m =>
            m[0] === selectedSquare.row &&
            m[1] === selectedSquare.col &&
            m[2] === row &&
            m[3] === col
        );
        if (isLegalMove) {
            if (piece !== EMPTY) {
                square.classList.add('legal-capture');
            } else {
                square.classList.add('legal-move');
            }
        }
    }

    square.addEventListener('click', () => onSquareClick(row, col, piece));
    boardEl.appendChild(square);
}

/**
 * Handle square click
 */
async function onSquareClick(row, col, piece) {
    if (isThinking) return;

    // Check if it's human turn
    const currentTurnColor = (currentBoard && currentBoard.length) ?
        // We need to track turn locally or infer from board?
        // App doesn't track global 'turn' variable well, inferred from status updates.
        // Let's rely on `legalMoves`. If legalMoves is empty, we can't move.
        // Also check if current player is configured as Human.
        'unknown' : 'unknown';

    // Simplified: If legalMoves is empty, we probably can't move.
    // Also, we should only allow selecting OUR pieces.
    // Logic: `isOurPiece` checks `playerColor`.
    // But now `playerColor` is just view perspective.
    // We need to check if the piece matches the Configured Human Color.

    // Determine whose turn it is? 
    // We don't have explicit Turn variable in global scope clearly updated.
    // Let's add `turn` to global state from updates.

    if (selectedSquare) {
        const move = [selectedSquare.row, selectedSquare.col, row, col];

        // Check if this is a legal move
        const isLegal = legalMoves.some(m =>
            m[0] === move[0] && m[1] === move[1] && m[2] === move[2] && m[3] === move[3]
        );

        if (isLegal) {
            await makeMove(move);
            selectedSquare = null;
        } else if (isPieceClickable(piece)) {
            // Select new piece
            selectedSquare = { row, col };
        } else {
            // Deselect
            selectedSquare = null;
        }
    } else {
        // Select piece if it's ours and we are human
        if (isPieceClickable(piece)) {
            selectedSquare = { row, col };
        }
    }

    renderBoard();
}

/**
 * Check if a piece belongs to the current turn's human player
 */
function isPieceClickable(piece) {
    // We technically need to know whose turn it is.
    // But simpler: Is this piece consistent with a HUMAN player configuration?
    // And is it their turn (implied by legalMoves existence for that color?)

    // If piece is WHITE, and White Config is Human.
    if (piece === WHITE && window.gameConfig && window.gameConfig.white === 'human') return true;
    if (piece === BLACK && window.gameConfig && window.gameConfig.black === 'human') return true;

    return false;
}

/**
 * Make a move
 */
async function makeMove(move) {
    isThinking = true;
    setStatus('🤔 Bot is thinking...'); // Default message, updated later
    statusEl.classList.add('thinking');

    try {
        const response = await fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ move })
        });

        const data = await response.json();

        if (data.error) {
            setStatus(`Error: ${data.error}`);
            isThinking = false;
            statusEl.classList.remove('thinking');
            return;
        }

        // Update state
        addMove(move, data.turn === 'black' ? 'white' : 'black'); // Previous turn moved
        currentBoard = data.board;
        legalMoves = data.legal_moves || [];

        if (data.bot_move) {
            addMove(data.bot_move, data.turn); // "Bot" label is generic, maybe improve?
            updateEval(data.evaluation);
        }

        renderBoard();
        updateStatus(data);

    } catch (error) {
        console.error('Error making move:', error);
        setStatus('Error communicating with server');
    }

    isThinking = false;
    statusEl.classList.remove('thinking');
}

/**
 * Loop for AI vs AI / Baseline vs AI
 */
async function gameLoop() {
    while (isAiVsAi && !stopAiFlag) {
        const statusResponse = await fetch('/api/state');
        const state = await statusResponse.json();

        if (state.game_over) {
            isAiVsAi = false;
            break;
        }

        isThinking = true;
        setStatus(`Auto-play: ${state.turn} moving...`);
        statusEl.classList.add('thinking');

        try {
            const response = await fetch('/api/bot_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();

            if (data.error) {
                console.error('Game Loop Error:', data.error);
                isAiVsAi = false;
                break;
            }

            // Update state
            currentBoard = data.board;
            legalMoves = data.legal_moves || [];

            if (data.bot_move) {
                // Determine who moved?
                // If turn WAS white, White moved.
                const whoMoved = state.turn;
                addMove(data.bot_move, whoMoved);
                updateEval(data.evaluation);
            }

            renderBoard();
            updateStatus(data);

            // Wait a bit
            await new Promise(resolve => setTimeout(resolve, 500));

        } catch (error) {
            console.error('Loop Network Error:', error);
            isAiVsAi = false;
            break;
        } finally {
            isThinking = false;
            statusEl.classList.remove('thinking');
        }
    }

    resetControls();
    if (stopAiFlag) {
        setStatus('Stopped play.');
        stopAiFlag = false;
    }
}

/**
 * Update the status display
 */
function updateStatus(data) {
    if (data.game_over) {
        setStatus(`Game Over: ${data.result}`);
    } else {
        const turn = data.turn === 'white' ? 'White' : 'Black';
        setStatus(`${turn} to move...`);
    }
}

function setStatus(text) {
    statusEl.textContent = text;
}

/**
 * Format a move for display
 */
function formatMove(move) {
    const files = 'abcdefgh';
    const fromSquare = files[move[1]] + (move[0] + 1);
    const toSquare = files[move[3]] + (move[2] + 1);
    return `${fromSquare}-${toSquare}`;
}

/**
 * Add move to history
 */
function addMove(move, player) {
    moveCount++;
    // Player argument: 'white', 'black', 'bot'(mapped to turn?)
    // Clean up labels:
    let label = player;
    if (player === 'bot') {
        // Infer from last known turn? Or just "Bot"
        // In mixed mode "Bot" is confusing.
        // Let's try to pass accurate "white" or "black" string.
        label = (moveCount % 2 === 1) ? 'White' : 'Black';
    } else if (player === 'player') {
        label = (moveCount % 2 === 1) ? 'White' : 'Black';
    } else {
        label = player.charAt(0).toUpperCase() + player.slice(1);
    }

    moveHistory.push({ move, player: label, number: moveCount });
    renderMoveHistory();
}

function renderMoveHistory() {
    movesEl.innerHTML = '';

    for (const entry of moveHistory) {
        const moveEl = document.createElement('div');
        moveEl.className = 'move-entry';
        const formatted = formatMove(entry.move);
        moveEl.innerHTML = `<span class="move-number">${entry.number}.</span> ${entry.player}: ${formatted}`;
        movesEl.appendChild(moveEl);
    }

    movesEl.scrollTop = movesEl.scrollHeight;
}

/**
 * Update evaluation bar
 */
function updateEval(value) {
    // Check if value is null (baseline)
    if (value === null || value === undefined) {
        const evalBar = document.querySelector('.eval-bar');
        evalBar.style.display = 'none'; // Hide bar
        evalLabelEl.textContent = '---';
        return;
    }

    // Show bar if receiving value
    const evalBar = document.querySelector('.eval-bar');
    evalBar.style.display = 'block';

    // Value is from -1 (black winning) to 1 (white winning)
    // Convert to percentage for white
    const percentage = ((value + 1) / 2) * 100;

    // Set CSS variable
    evalBar.style.setProperty('--eval-percent', `${percentage}%`);

    // Format label
    // If playing Black perspective, should we flip sign?
    // "Value" is usually absolute (positive=White adv).
    // The previous code flipped display if playerColor=='white'.
    // Let's just show absolute White Advantage (+/-).

    const displayValue = value;
    evalLabelEl.textContent = displayValue >= 0 ? `+${displayValue.toFixed(2)}` : displayValue.toFixed(2);
}

/**
 * Initialize with empty board display
 */
function initializeEmptyBoard() {
    // Create initial board state for display
    currentBoard = [];
    for (let row = 0; row < BOARD_SIZE; row++) {
        const rowArr = [];
        for (let col = 0; col < BOARD_SIZE; col++) {
            if (row < 2) {
                rowArr.push(WHITE);
            } else if (row >= BOARD_SIZE - 2) {
                rowArr.push(BLACK);
            } else {
                rowArr.push(EMPTY);
            }
        }
        currentBoard.push(rowArr);
    }
    renderBoard();
}

