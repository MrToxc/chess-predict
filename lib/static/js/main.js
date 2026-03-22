// Initializing Chessboard and Logic
let board = null;
const game = new Chess();
const $status = $('#game-status');
const $aiCommentary = $('#ai-commentary');

// UI Elements for Evaluation Bar
const $barWhite = $('#bar-white');
const $barDraw = $('#bar-draw');
const $barBlack = $('#bar-black');
const $valWhite = $('#val-white');
const $valDraw = $('#val-draw');
const $valBlack = $('#val-black');

// Move validation (prevents illegal moves)
function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;

    // User can only move pieces that are on turn
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
}

// Move processing after piece is dropped
function onDrop(source, target) {
    // Move creation
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' // Always promote to queen to simplify UI
    });

    // Invalid move
    if (move === null) return 'snapback';

    // Update UI and API request
    updateStatus();
    fetchAIPrediction();
}

// Animation of incoming move (e.g. after snapback)
function onSnapEnd() {
    board.position(game.fen());
}

// Textual status update
function updateStatus() {
    let status = '';
    let moveColor = game.turn() === 'w' ? 'White' : 'Black';

    if (game.in_checkmate()) {
        let winner = game.turn() === 'w' ? 'Black' : 'White';
        status = `Game over, ${winner} won!`;
        if (winner === 'White') {
            updateEvalBar(100, 0, 0);
        } else {
            updateEvalBar(0, 0, 100);
        }
    } else if (game.in_draw()) {
        status = 'Game ended in a draw.';
        updateEvalBar(0, 100, 0);
    } else {
        status = `${moveColor} is on turn.`;
        if (game.in_check()) {
            status += ` (Check!)`;
        }
    }
    
    $status.html(status);
}

// Communication with Python Backend
async function fetchAIPrediction() {
    // If game is over, no need to call AI (bar is already 100%)
    if (game.game_over()) return;

    const pgn = game.pgn();
    
    // If PGN is empty (start of game), no need to call API
    if (pgn === '') {
        updateEvalBar(50, 0, 50);
        return;
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pgn: pgn })
        });

        const data = await response.json();

        if (data.success) {
            const p = data.probabilities;
            updateEvalBar(p.white, p.draw, p.black);
        } else {
            console.error("API Error:", data.error);
        }
    } catch (err) {
        console.error("Fetch Error:", err);
    }
}

// Smooth update of percentages and bar widths
function updateEvalBar(white, draw, black) {
    // Width adjustment
    $barWhite.css('width', `${white}%`);
    $barDraw.css('width', `${draw}%`);
    $barBlack.css('width', `${black}%`);

    // Show text only where there is space (> 5%)
    $valWhite.text(white > 5 ? `${white.toFixed(1)}%` : '');
    $valDraw.text(draw > 5 ? `${draw.toFixed(1)}%` : '');
    $valBlack.text(black > 5 ? `${black.toFixed(1)}%` : '');
}

// Generating smart commentary based on numbers
function generateCommentary(w, b, d) {
    if (w > 85) return $aiCommentary.text("Crushing advantage for White. It's just a matter of technique now.");
    if (b > 85) return $aiCommentary.text("Crushing advantage for Black. White's defense is collapsing.");
    if (w > 65) return $aiCommentary.text("White has a solid advantage, controls the game and applies pressure.");
    if (b > 65) return $aiCommentary.text("Black is doing very well and taking the initiative.");
    if (d > 30) return $aiCommentary.text("The position is very drawish. Playing for minor nuances.");
    
    $aiCommentary.text("Even match. Every small tactical advantage can decide the game now.");
}

// Initialization
const config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
    pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
};

board = Chessboard('board', config);
updateStatus();

// Buttons
$('#resetBtn').on('click', () => {
    game.reset();
    board.start();
    updateStatus();
    updateEvalBar(50, 0, 50);
});

$('#undoBtn').on('click', () => {
    game.undo();
    board.position(game.fen());
    updateStatus();
    fetchAIPrediction();
});
