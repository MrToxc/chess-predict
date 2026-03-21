// Iniciaizace Sachovnice a Logiky
let board = null;
const game = new Chess();
const $status = $('#game-status');
const $aiCommentary = $('#ai-commentary');

// UI Elements pro Evaluation Bar
const $barWhite = $('#bar-white');
const $barDraw = $('#bar-draw');
const $barBlack = $('#bar-black');
const $valWhite = $('#val-white');
const $valDraw = $('#val-draw');
const $valBlack = $('#val-black');

// Validace tahu (zabrani ilegalnim tahum)
function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;

    // Uzivatel muze tahat jen figurkami, ktere jsou na tahu
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
}

// Zpracovani tahu po upusteni figurky
function onDrop(source, target) {
    // Vytvoreni tahu
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' // Vzdy menit na damu pro zjednoduseni UI
    });

    // Neplatny tah
    if (move === null) return 'snapback';

    // Aktualizace UI a dotaz na API
    updateStatus();
    fetchAIPrediction();
}

// Animace prichodiho tahu (treba po snapback)
function onSnapEnd() {
    board.position(game.fen());
}

// Aktualizace textoveho statusu
function updateStatus() {
    let status = '';
    let moveColor = game.turn() === 'w' ? 'Bílý' : 'Černý';

    if (game.in_checkmate()) {
        let winner = game.turn() === 'w' ? 'Černý' : 'Bílý';
        status = `Hra skončila, ${winner} vyhrál!`;
        if (winner === 'Bílý') {
            updateEvalBar(100, 0, 0);
        } else {
            updateEvalBar(0, 0, 100);
        }
    } else if (game.in_draw()) {
        status = 'Hra skončila remízou.';
        updateEvalBar(0, 100, 0);
    } else {
        status = `${moveColor} je na tahu.`;
        if (game.in_check()) {
            status += ` (Šach!)`;
        }
    }
    
    $status.html(status);
}

// Komunikace s Python Backendem
async function fetchAIPrediction() {
    // Pokud hra skoncila, neni treba volat AI (bar uz je 100%)
    if (game.game_over()) return;

    const pgn = game.pgn();
    
    // Pokud je PGN prazdne (start hry), netreba volat API
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

// Plynuly update procent a sirek baru
function updateEvalBar(white, draw, black) {
    // Nastaveni sirek
    $barWhite.css('width', `${white}%`);
    $barDraw.css('width', `${draw}%`);
    $barBlack.css('width', `${black}%`);

    // Zobrazeni jen textu, kde je misto (> 5%)
    $valWhite.text(white > 5 ? `${white.toFixed(1)}%` : '');
    $valDraw.text(draw > 5 ? `${draw.toFixed(1)}%` : '');
    $valBlack.text(black > 5 ? `${black.toFixed(1)}%` : '');
}

// Generovani chytreho komentare na zaklade cisel
function generateCommentary(w, b, d) {
    if (w > 85) return $aiCommentary.text("Drtivá převaha bílého. Tady už je to jen otázka techniky.");
    if (b > 85) return $aiCommentary.text("Drtivá převaha černého. Obrana bílého se hroutí.");
    if (w > 65) return $aiCommentary.text("Bílý má solidní výhodu, kontroluje hru a vyvíjí tlak.");
    if (b > 65) return $aiCommentary.text("Černý stojí velmi dobře a přebírá iniciativu.");
    if (d > 30) return $aiCommentary.text("Pozice je velmi remízová. Hraje se o drobné nuance.");
    
    $aiCommentary.text("Vyrovnaný zápas. Každá malá taktická výhoda teď může rozhodnout.");
}

// Inicializace
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

// Tlacitka
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
