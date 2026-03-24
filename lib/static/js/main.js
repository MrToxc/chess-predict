// Initializing Chessboard and Logic
let board = null;
const game = new Chess();
const $status = $('#game-status');
const $aiCommentary = $('#ai-commentary');

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
    let moveColor = game.turn() === 'w' ? 'Bílý' : 'Černý';

    if (game.in_checkmate()) {
        let winner = game.turn() === 'w' ? 'Černý' : 'Bílý';
        status = `Konec hry, ${winner} vyhrál!`;
    } else if (game.in_draw()) {
        status = 'Hra skončila remízou.';
    } else {
        status = `${moveColor} je na tahu.`;
        if (game.in_check()) {
            status += ` (Šach!)`;
        }
    }
    
    $status.html(status);
}

// Communication with Python Backend
async function fetchAIPrediction() {
    // If game is over, no need to call AI
    if (game.game_over()) {
        removeGhostPiece();
        return;
    }

    const pgn = game.pgn();
    
    // If PGN is empty (start of game), wait for first move or suggest opening
    if (pgn === '') {
        // We can still ask API for move 1
    }

    // Check if toggle is on
    const isAiEnabled = $('#aiToggle').is(':checked');
    if (!isAiEnabled) {
        removeGhostPiece();
        return;
    }

    $aiCommentary.text("AI přemýšlí nad dalšími tahy...");

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pgn: pgn })
        });

        const data = await response.json();

        if (data.success && data.best_move) {
            showGhostPiece(data.best_move, data.probability);
        } else {
            removeGhostPiece();
            $aiCommentary.text("AI nemá dostatek trénovacích dat pro tuto pozici.");
        }
    } catch (err) {
        console.error("Fetch Error:", err);
        $aiCommentary.text("Chyba při komunikaci s AI modelem.");
    }
}

function showGhostPiece(moveUci, probability) {
    removeGhostPiece();
    
    // target and source squares (e.g., e2e4)
    const fromSq = moveUci.substring(0, 2);
    const toSq = moveUci.substring(2, 4);
    
    // Highlight the target and source squares
    $(`.square-${toSq}`).addClass('highlight-ai');
    $(`.square-${fromSq}`).addClass('highlight-ai');
    
    // Draw ghost piece on target square
    const pieceObj = game.get(fromSq);
    if (pieceObj) {
        const pieceCode = pieceObj.color + pieceObj.type.toUpperCase(); // e.g. "wP" or "bK"
        const imgUrl = `https://chessboardjs.com/img/chesspieces/wikipedia/${pieceCode}.png`;
        const ghostImg = `<img src="${imgUrl}" class="ai-ghost-piece" style="width: 100%; height: 100%; position: absolute; z-index: 2; pointer-events: none;">`;
        $(`.square-${toSq}`).append(ghostImg);
    }
    
    $aiCommentary.text(`AI Navrhuje: ${moveUci} (Jistota modelu: ${probability.toFixed(1)}%)`);
}

function removeGhostPiece() {
    $('.square-55d63').removeClass('highlight-ai');
    $('.ai-ghost-piece').remove();
    $aiCommentary.text('');
}

// Toggle Listneer
$('#aiToggle').on('change', function() {
    if (this.checked) {
        fetchAIPrediction();
    } else {
        removeGhostPiece();
    }
});

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
if ($('#aiToggle').is(':checked')) {
    fetchAIPrediction();
}

// Buttons
$('#resetBtn').on('click', () => {
    game.reset();
    board.start();
    updateStatus();
    removeGhostPiece();
    if ($('#aiToggle').is(':checked')) {
        fetchAIPrediction();
    }
});

$('#undoBtn').on('click', () => {
    game.undo();
    board.position(game.fen());
    updateStatus();
    fetchAIPrediction();
});
