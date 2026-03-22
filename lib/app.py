import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from extractor import extract_features

app = Flask(__name__)
CORS(app)

# Path configuration (assuming app.py runs from the lib/ directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'trained_model.pkl')
SCALER_PATH = os.path.join(current_dir, 'scaler.pkl')
LE_PATH = os.path.join(current_dir, 'label_encoder.pkl')
FEATURES_PATH = os.path.join(current_dir, 'input_features.pkl')

print("Loading models into memory...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LE_PATH)
    input_features = joblib.load(FEATURES_PATH)
    print("Models successfully loaded.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Run `python model.py` in the main directory first.")
    model, scaler, label_encoder, input_features = None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not scaler or not input_features or not label_encoder:
        return jsonify({"error": "Models are not loaded. Run the training first."}), 500

    data = request.json
    pgn_string = data.get('pgn', '')
    
    if not pgn_string:
        # Initial position has no moves, odds are even
        return jsonify({
            "success": True,
            "probabilities": {
                "white": 50.0,
                "draw": 0.0,
                "black": 50.0
            }
        })

    # Feature extraction using our extractor
    # We pass "*" as the result because it doesn't matter for prediction
    # For live web prediction we don't hide moves, we want exactly this move (random_subset=False)
    features_dict = extract_features(pgn_string, "*", random_subset=False)
    
    if not features_dict:
        return jsonify({"error": "Failed to extract features from the board position."}), 400

    # Preparing data for the model
    # We must maintain the exact feature order the model expects
    prediction_df = pd.DataFrame([features_dict], columns=input_features)
    
    # Scaling (Standardization)
    X_std = scaler.transform(prediction_df)
    
    # Prediction
    probabilities = model.predict_proba(X_std)[0]
    
    # Mapping probabilities from indices to logical categories
    classes = list(label_encoder.classes_)
    
    result = {
        "white": probabilities[classes.index('1-0')] * 100,
        "draw": probabilities[classes.index('1/2-1/2')] * 100,
        "black": probabilities[classes.index('0-1')] * 100
    }

    return jsonify({
        "success": True,
        "probabilities": result,
        "features_extracted": len(input_features)
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
