import os
import joblib
import pandas as pd
import chess.pgn
from io import StringIO
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from extractor import (
    get_base_stats, 
    get_incremental_move_stats, 
    extract_single_state_features, 
    parse_eco_category,
    EMPTY_FEATURES,
    HISTORY_LENGTH
)

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'trained_model.pkl')
SCALER_PATH = os.path.join(current_dir, 'scaler.pkl')
FEATURES_PATH = os.path.join(current_dir, 'input_features.pkl')

print("Loading models into memory...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    input_features = joblib.load(FEATURES_PATH)
    print("Models successfully loaded.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Run `python model.py` in the main directory first.")
    model, scaler, input_features = None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not scaler or not input_features:
        return jsonify({"error": "Models are not loaded."}), 500

    data = request.json
    pgn_string = data.get('pgn', '')
    
    if not pgn_string:
        return jsonify({"success": True, "best_move": None, "probability": 0})

    try:
        game = chess.pgn.read_game(StringIO(pgn_string))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if game is None:
        return jsonify({"error": "Could not parse PGN."}), 400

    eco_category = parse_eco_category(game.headers)
    white_elo = 1500  # Default fallback 
    black_elo = 1500

    board = game.board()
    moves = list(game.mainline())
    total_half_moves = len(moves)

    # 1. Replay game to get history states and current board
    history_states = []
    current_stats = get_base_stats()

    for half_move_index, node in enumerate(moves):
        is_white = (half_move_index % 2 == 0)
        full_move_number = (half_move_index // 2) + 1
        
        state_features = extract_single_state_features(board, half_move_index, current_stats, eco_category)
        history_states.append(state_features)
        
        current_stats = get_incremental_move_stats(board, node.move, current_stats, is_white, full_move_number)
        board.push(node.move)

    if board.is_game_over():
        return jsonify({"success": True, "best_move": None, "probability": 0})

    # 2. Generate candidate rows
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return jsonify({"success": True, "best_move": None, "probability": 0})
        
    is_w = (total_half_moves % 2 == 0)
    f_mn = (total_half_moves // 2) + 1
    
    rows = []
    
    for candidate_move in legal_moves:
        cand_stats = get_incremental_move_stats(board, candidate_move, current_stats, is_w, f_mn)
        board.push(candidate_move)
        cand_features = extract_single_state_features(board, total_half_moves + 1, cand_stats, eco_category)
        board.pop()
        
        row = {
            "turn_index": total_half_moves,
            "white_elo": white_elo,
            "black_elo": black_elo,
        }
        
        for h in range(1, HISTORY_LENGTH + 1):
            hist_idx = total_half_moves - (HISTORY_LENGTH - h + 1)
            hist_f = EMPTY_FEATURES if hist_idx < 0 else history_states[hist_idx]
            for key, value in hist_f.items():
                row[f"hist_{h}_{key}"] = value
                
        for key, value in cand_features.items():
            row[f"cand_{key}"] = value
            
        rows.append(row)

    # Re-order columns strictly safely to match precisely the model
    df = pd.DataFrame(rows)
    try:
        df = df[input_features]
    except KeyError as e:
        return jsonify({"error": f"Model mismatch on features. Missing: {e}"}), 500

    X_std = scaler.transform(df)
    
    # model.predict_proba returns array shape (n_samples, 2), index 1 is "played"
    probs = model.predict_proba(X_std)[:, 1]
    
    best_index = probs.argmax()
    best_move = legal_moves[best_index]
    best_prob = probs[best_index]

    return jsonify({
        "success": True,
        "best_move": best_move.uci(), 
        "probability": float(best_prob * 100),
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
