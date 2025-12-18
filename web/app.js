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
    document.getElementById('newGameWhite').addEventListener('click', () => startGame('white'));
    document.getElementById('newGameBlack').addEventListener('click', () => startGame('black'));

    // Initial render
    initializeEmptyBoard();
}

init();

/**
 * Start a new game
 */
async function startGame(color) {
    playerColor = color;
    moveHistory = [];
    selectedSquare = null;
    moveCount = 0;

    // Get selected model size
    const sizeSelect = document.getElementById('modelSize');
    const modelSize = sizeSelect ? sizeSelect.value : 'medium';

    setStatus(`Loading ${modelSize} model...`);

    try {
        const response = await fetch('/api/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ color, size: modelSize })
        });

        const data = await response.json();
        currentBoard = data.board;
        legalMoves = data.legal_moves || [];

        if (data.bot_move) {
            addMove(data.bot_move, 'bot');
            updateEval(data.evaluation || 0);
        }

        renderBoard();
        updateStatus(data);

    } catch (error) {
        console.error('Error starting game:', error);
        setStatus('Error starting game. Is the server running?');
    }

    // Update eval bar orientation
    const evalBar = document.querySelector('.eval-bar');
    if (color === 'black') {
        evalBar.classList.add('flipped');
    } else {
        evalBar.classList.remove('flipped');
    }
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

    // If we have a selected piece, try to move
    if (selectedSquare) {
        const move = [selectedSquare.row, selectedSquare.col, row, col];

        // Check if this is a legal move
        const isLegal = legalMoves.some(m =>
            m[0] === move[0] && m[1] === move[1] && m[2] === move[2] && m[3] === move[3]
        );

        if (isLegal) {
            await makeMove(move);
            selectedSquare = null;
        } else if (isOurPiece(piece)) {
            // Select new piece
            selectedSquare = { row, col };
        } else {
            // Deselect
            selectedSquare = null;
        }
    } else {
        // Select piece if it's ours
        if (isOurPiece(piece)) {
            selectedSquare = { row, col };
        }
    }

    renderBoard();
}

/**
 * Check if a piece belongs to the current player
 */
function isOurPiece(piece) {
    if (playerColor === 'white') {
        return piece === WHITE;
    } else {
        return piece === BLACK;
    }
}

/**
 * Make a move
 */
async function makeMove(move) {
    isThinking = true;
    setStatus('🤔 Bot is thinking...');
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
        addMove(move, 'player');
        currentBoard = data.board;
        legalMoves = data.legal_moves || [];

        if (data.bot_move) {
            addMove(data.bot_move, 'bot');
            updateEval(data.evaluation || 0);
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
 * Update the status display
 */
function updateStatus(data) {
    if (data.game_over) {
        setStatus(`Game Over: ${data.result}`);
    } else {
        const turn = data.turn === 'white' ? 'White' : 'Black';
        const yourTurn = (playerColor === data.turn);
        setStatus(yourTurn ? `Your turn (${turn})` : `Bot's turn (${turn})`);
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
    moveHistory.push({ move, player, number: moveCount });
    renderMoveHistory();
}

function renderMoveHistory() {
    movesEl.innerHTML = '';

    for (const entry of moveHistory) {
        const moveEl = document.createElement('div');
        moveEl.className = 'move-entry';
        const formatted = formatMove(entry.move);
        const playerLabel = entry.player === 'player' ? 'You' : 'Bot';
        moveEl.innerHTML = `<span class="move-number">${entry.number}.</span> ${playerLabel}: ${formatted}`;
        movesEl.appendChild(moveEl);
    }

    movesEl.scrollTop = movesEl.scrollHeight;
}

/**
 * Update evaluation bar
 */
function updateEval(value) {
    // Value is from -1 (black winning) to 1 (white winning)
    // Convert to percentage for white
    const percentage = ((value + 1) / 2) * 100;

    // Set CSS variable
    const evalBar = document.querySelector('.eval-bar');
    evalBar.style.setProperty('--eval-percent', `${percentage}%`);

    // Format label
    // If playing white, show value as is. If playing black, negate it.
    const displayValue = playerColor === 'white' ? value : -value;
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

