import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from extractor import extract_features

app = Flask(__name__)
CORS(app)

# Nastaveni cest (predpokladame, ze app.py bezi z adresare lib/)
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'trained_model.pkl')
SCALER_PATH = os.path.join(current_dir, 'scaler.pkl')
LE_PATH = os.path.join(current_dir, 'label_encoder.pkl')
FEATURES_PATH = os.path.join(current_dir, 'input_features.pkl')

print("Nacitam modely do pameti...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LE_PATH)
    input_features = joblib.load(FEATURES_PATH)
    print("Modely uspesne nacteny.")
except Exception as e:
    print(f"Chyba pri nacitani modelu: {e}")
    print("Spustte nejprve `python model.py` v hlavnim adresari.")
    model, scaler, label_encoder, input_features = None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not scaler or not input_features or not label_encoder:
        return jsonify({"error": "Modely nejsou nacteny. Spustte nejprve trenink."}), 500

    data = request.json
    pgn_string = data.get('pgn', '')
    
    if not pgn_string:
        # Pocatecni pozice nema zadne tahy, sance jsou vyrovnane
        return jsonify({
            "success": True,
            "probabilities": {
                "white": 50.0,
                "draw": 0.0,
                "black": 50.0
            }
        })

    # Extrakce atributu pomoci naseho extraktoru
    # Predame "*" jako vysledek, protoze ten neni pro predikci dulezity
    # Pro live webovou predikci neschovavame tahy, ale chceme presne tento tah (random_subset=False)
    features_dict = extract_features(pgn_string, "*", random_subset=False)
    
    if not features_dict:
        return jsonify({"error": "Nepodarilo se extrahovat atributy z hraciho pole."}), 400

    # Priprava dat pro model
    # Musime zachovat presne poradi atributu, kere model ocekava
    prediction_df = pd.DataFrame([features_dict], columns=input_features)
    
    # Skalovani (Standardizace)
    X_std = scaler.transform(prediction_df)
    
    # Predikce
    probabilities = model.predict_proba(X_std)[0]
    
    # Mapovani pobi z indexu do logickych kategorii
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
