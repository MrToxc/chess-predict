"""Enhanced chess game feature extractor with deeper positional analysis."""

import json
import os
import logging
import random
from io import StringIO

import chess
import chess.pgn
import pandas as pd

INPUT_FILE_PATH = os.path.join("data", "raw_games.json")
OUTPUT_FILE_PATH = os.path.join("data", "features.csv")
PROGRESS_FILE_PATH = os.path.join("data", "extractor_progress.json")

# Configuration for sequential modeling
HISTORY_LENGTH = 3
SAMPLES_PER_GAME = 3
MAX_GAMES_TO_PROCESS = None  # Set to an integer to limit the games processed (e.g., 1000)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

WHITE_MINOR_PIECE_STARTING_SQUARES = {chess.B1, chess.G1, chess.C1, chess.F1}
BLACK_MINOR_PIECE_STARTING_SQUARES = {chess.B8, chess.G8, chess.C8, chess.F8}
CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]
EXTENDED_CENTER_SQUARES = [chess.C3, chess.D3, chess.E3, chess.F3,
                           chess.C4, chess.D4, chess.E4, chess.F4,
                           chess.C5, chess.D5, chess.E5, chess.F5,
                           chess.C6, chess.D6, chess.E6, chess.F6]
VALID_RESULTS = ("1-0", "0-1", "1/2-1/2", "*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Board analysis functions
# ---------------------------------------------------------------------------

def compute_material_difference(board: chess.Board) -> int:
    difference = 0
    for piece_type, value in PIECE_VALUES.items():
        difference += len(board.pieces(piece_type, chess.WHITE)) * value
        difference -= len(board.pieces(piece_type, chess.BLACK)) * value
    return difference


def compute_total_material(board: chess.Board) -> tuple[int, int]:
    white_material = 0
    black_material = 0
    for piece_type, value in PIECE_VALUES.items():
        white_material += len(board.pieces(piece_type, chess.WHITE)) * value
        black_material += len(board.pieces(piece_type, chess.BLACK)) * value
    return white_material, black_material


def count_pieces_by_type(board: chess.Board) -> dict:
    return {
        "white_pawns": len(board.pieces(chess.PAWN, chess.WHITE)),
        "black_pawns": len(board.pieces(chess.PAWN, chess.BLACK)),
        "white_knights": len(board.pieces(chess.KNIGHT, chess.WHITE)),
        "black_knights": len(board.pieces(chess.KNIGHT, chess.BLACK)),
        "white_bishops": len(board.pieces(chess.BISHOP, chess.WHITE)),
        "black_bishops": len(board.pieces(chess.BISHOP, chess.BLACK)),
        "white_rooks": len(board.pieces(chess.ROOK, chess.WHITE)),
        "black_rooks": len(board.pieces(chess.ROOK, chess.BLACK)),
        "white_queens": len(board.pieces(chess.QUEEN, chess.WHITE)),
        "black_queens": len(board.pieces(chess.QUEEN, chess.BLACK)),
    }


def has_bishop_pair(board: chess.Board, color: chess.Color) -> int:
    return 1 if len(board.pieces(chess.BISHOP, color)) >= 2 else 0


def count_developed_pieces(board: chess.Board, color: chess.Color, starting_squares: set[int]) -> int:
    count = 0
    for square in starting_squares:
        piece = board.piece_at(square)
        is_original = (piece is not None and piece.color == color
                       and piece.piece_type in (chess.KNIGHT, chess.BISHOP))
        if not is_original:
            count += 1
    return count


def compute_center_control(board: chess.Board) -> tuple[int, int]:
    white_control = sum(1 for sq in CENTER_SQUARES if board.is_attacked_by(chess.WHITE, sq))
    black_control = sum(1 for sq in CENTER_SQUARES if board.is_attacked_by(chess.BLACK, sq))
    return white_control, black_control


def compute_extended_center_control(board: chess.Board) -> tuple[int, int]:
    white_control = sum(1 for sq in EXTENDED_CENTER_SQUARES if board.is_attacked_by(chess.WHITE, sq))
    black_control = sum(1 for sq in EXTENDED_CENTER_SQUARES if board.is_attacked_by(chess.BLACK, sq))
    return white_control, black_control


def compute_mobility(board: chess.Board) -> tuple[int, int]:
    """Count legal moves for the side to move, then flip to count the other side."""
    if board.turn == chess.WHITE:
        white_mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())
        black_mobility = len(list(board.legal_moves))
        board.pop()
    else:
        black_mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())
        white_mobility = len(list(board.legal_moves))
        board.pop()
    return white_mobility, black_mobility


def count_attacked_squares(board: chess.Board) -> tuple[int, int]:
    white_attacks = 0
    black_attacks = 0
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            white_attacks += 1
        if board.is_attacked_by(chess.BLACK, square):
            black_attacks += 1
    return white_attacks, black_attacks


def count_pawn_structure_features(board: chess.Board, color: chess.Color) -> dict:
    """Count doubled, isolated, and passed pawns for a given color."""
    pawns = board.pieces(chess.PAWN, color)
    opponent_pawns = board.pieces(chess.PAWN, not color)

    doubled_count = 0
    isolated_count = 0
    passed_count = 0

    files_with_pawns = set()
    for pawn_square in pawns:
        files_with_pawns.add(chess.square_file(pawn_square))

    for pawn_square in pawns:
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)

        # Doubled: another own pawn on the same file
        same_file_pawns = [sq for sq in pawns if chess.square_file(sq) == pawn_file and sq != pawn_square]
        if same_file_pawns:
            doubled_count += 1

        # Isolated: no own pawns on adjacent files
        adjacent_files = []
        if pawn_file > 0:
            adjacent_files.append(pawn_file - 1)
        if pawn_file < 7:
            adjacent_files.append(pawn_file + 1)
        has_neighbor = any(f in files_with_pawns for f in adjacent_files)
        if not has_neighbor:
            isolated_count += 1

        # Passed: no opponent pawns on same or adjacent files ahead
        is_passed = True
        check_files = [pawn_file] + adjacent_files
        for opponent_square in opponent_pawns:
            opponent_file = chess.square_file(opponent_square)
            opponent_rank = chess.square_rank(opponent_square)
            if opponent_file not in check_files:
                continue
            if color == chess.WHITE and opponent_rank > pawn_rank:
                is_passed = False
                break
            if color == chess.BLACK and opponent_rank < pawn_rank:
                is_passed = False
                break
        if is_passed:
            passed_count += 1

    return {
        "doubled": doubled_count,
        "isolated": isolated_count,
        "passed": passed_count,
    }


def compute_king_safety(board: chess.Board, color: chess.Color) -> int:
    """Count how many pawns are on adjacent squares around the king."""
    king_square = board.king(color)
    if king_square is None:
        return 0

    pawn_shield = 0
    for adjacent_square in chess.SQUARES:
        distance = chess.square_distance(king_square, adjacent_square)
        if distance > 2:
            continue
        piece = board.piece_at(adjacent_square)
        if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
            pawn_shield += 1

    return pawn_shield


def compute_king_exposure(board: chess.Board, color: chess.Color) -> int:
    """Count how many squares around the king are attacked by the opponent."""
    king_square = board.king(color)
    if king_square is None:
        return 0

    opponent_color = not color
    attacked_count = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    for file_offset in range(-1, 2):
        for rank_offset in range(-1, 2):
            if file_offset == 0 and rank_offset == 0:
                continue
            target_file = king_file + file_offset
            target_rank = king_rank + rank_offset
            if not (0 <= target_file <= 7 and 0 <= target_rank <= 7):
                continue
            target_square = chess.square(target_file, target_rank)
            if board.is_attacked_by(opponent_color, target_square):
                attacked_count += 1

    return attacked_count


def parse_eco_category(headers: chess.pgn.Headers) -> int:
    eco_code = headers.get("ECO", "")
    if not eco_code:
        return -1
    first_letter = eco_code[0].upper()
    if first_letter not in "ABCDE":
        return -1
    return ord(first_letter) - ord("A")


def compute_hanging_material(board: chess.Board) -> tuple[int, int]:
    """Compute the total material value of undefended pieces attacked by the opponent."""
    white_hanging = 0
    black_hanging = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        color = piece.color
        opponent_color = not color

        if board.is_attacked_by(opponent_color, square):
            if not board.is_attacked_by(color, square):
                value = PIECE_VALUES.get(piece.piece_type, 0)
                if color == chess.WHITE:
                    white_hanging += value
                else:
                    black_hanging += value

    return white_hanging, black_hanging


# ---------------------------------------------------------------------------
# Move tracking and incremental stats
# ---------------------------------------------------------------------------

def get_base_stats() -> dict:
    return {
        "white_castled": 0,
        "black_castled": 0,
        "white_castled_on_move": 0,
        "black_castled_on_move": 0,
        "pawn_moves_white": 0,
        "pawn_moves_black": 0,
        "piece_moves_white": 0,
        "piece_moves_black": 0,
        "captures_white": 0,
        "captures_black": 0,
        "checks_white": 0,
        "checks_black": 0,
    }

def get_incremental_move_stats(board: chess.Board, move: chess.Move, base_stats: dict, is_white: bool, full_move_number: int) -> dict:
    new_stats = base_stats.copy()

    if board.is_castling(move):
        if is_white:
            new_stats["white_castled"] = 1
            new_stats["white_castled_on_move"] = full_move_number
        else:
            new_stats["black_castled"] = 1
            new_stats["black_castled_on_move"] = full_move_number

    if board.is_capture(move):
        if is_white:
            new_stats["captures_white"] += 1
        else:
            new_stats["captures_black"] += 1

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is not None:
        if moving_piece.piece_type == chess.PAWN:
            if is_white:
                new_stats["pawn_moves_white"] += 1
            else:
                new_stats["pawn_moves_black"] += 1
        else:
            if is_white:
                new_stats["piece_moves_white"] += 1
            else:
                new_stats["piece_moves_black"] += 1

    board.push(move)
    if board.is_check():
        if is_white:
            new_stats["checks_white"] += 1
        else:
            new_stats["checks_black"] += 1
    board.pop()

    return new_stats


def extract_single_state_features(board: chess.Board, completed_half_moves: int, move_stats: dict, eco_category: int) -> dict:
    """Extracts features for a single board state."""
    move_number = (completed_half_moves // 2) + 1

    # Position analysis
    material_diff = compute_material_difference(board)
    white_material, black_material = compute_total_material(board)
    piece_counts = count_pieces_by_type(board)

    white_developed = count_developed_pieces(board, chess.WHITE, WHITE_MINOR_PIECE_STARTING_SQUARES)
    black_developed = count_developed_pieces(board, chess.BLACK, BLACK_MINOR_PIECE_STARTING_SQUARES)

    center_white, center_black = compute_center_control(board)
    ext_center_white, ext_center_black = compute_extended_center_control(board)

    white_mobility, black_mobility = compute_mobility(board)
    white_attacks, black_attacks = count_attacked_squares(board)

    white_pawn_structure = count_pawn_structure_features(board, chess.WHITE)
    black_pawn_structure = count_pawn_structure_features(board, chess.BLACK)

    white_king_safety = compute_king_safety(board, chess.WHITE)
    black_king_safety = compute_king_safety(board, chess.BLACK)
    white_king_exposure = compute_king_exposure(board, chess.WHITE)
    black_king_exposure = compute_king_exposure(board, chess.BLACK)

    white_hanging, black_hanging = compute_hanging_material(board)

    features = {
        "move_number": move_number,
        "side_to_move": 1 if board.turn == chess.WHITE else 0,
        "material_diff": material_diff,
        "white_material": white_material,
        "black_material": black_material,
        **piece_counts,
        "white_bishop_pair": has_bishop_pair(board, chess.WHITE),
        "black_bishop_pair": has_bishop_pair(board, chess.BLACK),
        "white_developed": white_developed,
        "black_developed": black_developed,
        **move_stats,
        "center_control_white": center_white,
        "center_control_black": center_black,
        "ext_center_white": ext_center_white,
        "ext_center_black": ext_center_black,
        "white_mobility": white_mobility,
        "black_mobility": black_mobility,
        "white_attacked_squares": white_attacks,
        "black_attacked_squares": black_attacks,
        "white_doubled_pawns": white_pawn_structure["doubled"],
        "black_doubled_pawns": black_pawn_structure["doubled"],
        "white_isolated_pawns": white_pawn_structure["isolated"],
        "black_isolated_pawns": black_pawn_structure["isolated"],
        "white_passed_pawns": white_pawn_structure["passed"],
        "black_passed_pawns": black_pawn_structure["passed"],
        "white_king_safety": white_king_safety,
        "black_king_safety": black_king_safety,
        "white_king_exposure": white_king_exposure,
        "black_king_exposure": black_king_exposure,
        "white_hanging": white_hanging,
        "black_hanging": black_hanging,
        "eco_category": eco_category,
    }

    return features

def get_empty_features() -> dict:
    """Returns a dictionary of features with all values set to zero for history padding."""
    b = chess.Board()
    f = extract_single_state_features(b, 0, get_base_stats(), -1)
    for k in f.keys():
        f[k] = 0
    return f

EMPTY_FEATURES = get_empty_features()

# ---------------------------------------------------------------------------
# Main extraction sequence loop
# ---------------------------------------------------------------------------

def process_game(game_record: dict, game_index: int) -> list[dict]:
    pgn_string = game_record.get("pgn", "")
    if not pgn_string:
        return []

    try:
        game = chess.pgn.read_game(StringIO(pgn_string))
    except Exception as exception:
        logger.warning("Failed to parse PGN: %s", exception)
        return []

    if game is None:
        return []

    eco_category = parse_eco_category(game.headers)
    white_elo = game_record.get("white_rating", 1500)
    black_elo = game_record.get("black_rating", 1500)
    
    # We will enforce valid numbers just in case
    if white_elo is None: white_elo = 1500
    if black_elo is None: black_elo = 1500

    board = game.board()
    moves = list(game.mainline())
    total_half_moves = len(moves)

    if total_half_moves < 2:
        return []

    # 1. Collect all history states for the entire game
    history_states = []
    current_stats = get_base_stats()

    for half_move_index, node in enumerate(moves):
        is_white = (half_move_index % 2 == 0)
        full_move_number = (half_move_index // 2) + 1

        # Extract BEFORE the move is played
        state_features = extract_single_state_features(board, half_move_index, current_stats, eco_category)
        history_states.append(state_features)

        # Update board and stats
        current_stats = get_incremental_move_stats(board, node.move, current_stats, is_white, full_move_number)
        board.push(node.move)

    # 2. Pick random turns to sample
    safe_max_samples = min(SAMPLES_PER_GAME, total_half_moves)
    sample_indices = random.sample(range(total_half_moves), safe_max_samples)

    generated_rows = []

    for turn_idx in sample_indices:
        played_move = moves[turn_idx].move

        # Reconstruct the board up to this exact turn
        b = game.board()
        stats = get_base_stats()
        for i in range(turn_idx):
            is_w = (i % 2 == 0)
            f_mn = (i // 2) + 1
            stats = get_incremental_move_stats(b, moves[i].move, stats, is_w, f_mn)
            b.push(moves[i].move)

        legal_moves = list(b.legal_moves)
        is_w = (turn_idx % 2 == 0)
        f_mn = (turn_idx // 2) + 1

        # Generate a candidate row for EVERY legal move
        played_row = None
        unplayed_rows = []

        for candidate_move in legal_moves:
            # 1. Get candidate features
            cand_stats = get_incremental_move_stats(b, candidate_move, stats, is_w, f_mn)
            b.push(candidate_move)
            cand_features = extract_single_state_features(b, turn_idx + 1, cand_stats, eco_category)
            b.pop()

            was_played = 1 if candidate_move == played_move else 0

            # 2. Initialize the row with base info
            row = {
                "game_id": game_index,
                "turn_index": turn_idx,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "was_played": was_played
            }

            # 3. Add History parameters
            for h in range(1, HISTORY_LENGTH + 1):
                hist_idx = turn_idx - (HISTORY_LENGTH - h + 1)
                
                if hist_idx < 0:
                    hist_f = EMPTY_FEATURES
                else:
                    hist_f = history_states[hist_idx]

                for key, value in hist_f.items():
                    row[f"hist_{h}_{key}"] = value

            # 4. Add Candidate parameters
            for key, value in cand_features.items():
                row[f"cand_{key}"] = value

            if was_played == 1:
                played_row = row
            else:
                unplayed_rows.append(row)

        # 5. DOWNSAMPLING (The fix for the AI guessing bad moves)
        # Instead of feeding the AI 30 bad moves for every 1 good move (which makes it just guess '0' for everything),
        # we will only keep 3 random bad moves. This 1:3 ratio forces the Neural Network to actually learn chess patterns!
        if played_row:
            generated_rows.append(played_row)
        
        safe_sample_count = min(3, len(unplayed_rows))
        generated_rows.extend(random.sample(unplayed_rows, safe_sample_count))

    return generated_rows


def load_progress() -> int:
    if os.path.exists(PROGRESS_FILE_PATH):
        try:
            with open(PROGRESS_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_processed_game_index", -1)
        except Exception:
            return -1
    return -1

def save_progress(game_index: int) -> None:
    os.makedirs(os.path.dirname(PROGRESS_FILE_PATH), exist_ok=True)
    with open(PROGRESS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_processed_game_index": game_index}, f)

def run_extraction() -> None:
    if not os.path.exists(INPUT_FILE_PATH):
        logger.error("Input file not found: %s. Run crawler.py first.", INPUT_FILE_PATH)
        return

    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as file:
        raw_games = json.load(file)

    logger.info("Loaded %d raw games from %s", len(raw_games), INPUT_FILE_PATH)

    if MAX_GAMES_TO_PROCESS is not None:
        raw_games = raw_games[:MAX_GAMES_TO_PROCESS]
        logger.info("Limiting extraction to %d games.", len(raw_games))

    last_processed = load_progress()
    start_index = last_processed + 1

    if start_index >= len(raw_games):
        logger.info("All games have already been processed.")
        return

    if start_index > 0:
        logger.info("Resuming extraction from game index %d", start_index)

    batch_features = []
    skipped_count = 0
    error_count = 0
    total_extracted_in_run = 0

    for game_index in range(start_index, len(raw_games)):
        game_record = raw_games[game_index]
        try:
            game_rows = process_game(game_record, game_index)
            if not game_rows:
                skipped_count += 1
            else:
                batch_features.extend(game_rows)
        except Exception as exception:
            logger.warning("Error processing game %d: %s", game_index, exception)
            error_count += 1

        # Save batch every 500 games or at the end
        if (game_index + 1) % 500 == 0 or game_index == len(raw_games) - 1:
            if batch_features:
                dataframe = pd.DataFrame(batch_features)
                os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
                write_header = not os.path.exists(OUTPUT_FILE_PATH)
                dataframe.to_csv(OUTPUT_FILE_PATH, mode='a', header=write_header, index=False)
                total_extracted_in_run += len(batch_features)
                batch_features = []

            save_progress(game_index)
            logger.info("  Processed %d / %d games. Progress saved.", game_index + 1, len(raw_games))

    logger.info("Done. Extracted Rows in this run: %d | Skipped: %d | Errors: %d", total_extracted_in_run, skipped_count, error_count)


if __name__ == "__main__":
    run_extraction()
