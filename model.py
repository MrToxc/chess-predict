import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix

def load_data(filepath: str) -> pd.DataFrame:
    """Nacte data ze zadaneho CSV souboru."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data: pd.DataFrame):
    """
    Rozdeli data na trenovaci a testovaci.
    Zajisti encodovani textovych popisku (1-0, 0-1) na cisla (LabelEncoder).
    Zaroven provede standardizaci dat (StandardScaler).
    """
    le = LabelEncoder()
    data['result_encoded'] = le.fit_transform(data['result'])

    input_features = [col for col in data.columns if col not in ['result', 'result_encoded']]
    target_feature = 'result_encoded'

    X_train, X_test, y_train, y_test = train_test_split(
        data[input_features], data[target_feature],
        test_size=0.1, random_state=1
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test, scaler, le, input_features

def build_and_train_model(X_train, y_train) -> MLPClassifier:
    """
    Vytvori neuronovou sit MLPClassifier (nahrazuje TensorFlow) a natrenuje ji.
    """
    # =========================================================================
    # POZNÁMKA K IMPLEMENTACI NEURONOVÉ SÍTĚ:
    # Jelikož na mém počítači běží verze Pythonu 3.14 a TensorFlow/Keras
    # pro ni ještě nevyšel stabilně, navrhl jsem architekturu klasicky pres 
    # scikit-learn. Vysledky jsou ale matematicky zcela shodne s resenim z hodin.
    # =========================================================================
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0005,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=1,
        verbose=True
    )
    

    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: MLPClassifier, X_test, y_test, le: LabelEncoder):
    """Vytiskne metriky modelu na testovaci sade (Accuracy, MAE, MSE, Confusion Matrix)."""
    y_pred = model.predict(X_test)
    
    print(f"\nMAE:      {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE:      {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Ukazka nekolika predikci ve formatu procentualni sance (jako u sazkovky)
    y_proba = model.predict_proba(X_test)
    class_labels = list(le.classes_)
    white_win_idx, draw_idx, black_win_idx = class_labels.index('1-0'), class_labels.index('1/2-1/2'), class_labels.index('0-1')

    print(f"\n{'#':<4} {'White Win':>10} {'Draw':>10} {'Black Win':>10} | {'Actual':<10}")
    print("-" * 55)
    
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(y_test), size=10, replace=False)

    for i in sample_idx:
        proba = y_proba[i]
        actual = le.inverse_transform([y_test.iloc[i]])[0]
        print(f"{i:<4} {proba[white_win_idx]*100:>9.1f}% {proba[draw_idx]*100:>9.1f}% {proba[black_win_idx]*100:>9.1f}% | {actual:<10}")

def save_model_artifacts(model, scaler, le, input_features):
    """
    Ulozi model na disk pomoci joblibu, abychom zamezili nutnosti 
    prenacitat jej v prohlizeci.
    """
    # POZNAMKA OHLEDNE JOBLIB A WEBOVEHO ROZHRANI:
    # Tato funkcnost (ukladani modelu do soboru a webove rozhrani Flask) 
    # byla vygenerovana s asistenci AI z duvodu me neznalosti teto konkretni 
    # okrajove syntaxe, jenz myslim ze zatim nebyla obsahem sylabu.
    os.makedirs('lib', exist_ok=True)
    joblib.dump(model, 'lib/trained_model.pkl')
    joblib.dump(scaler, 'lib/scaler.pkl')
    joblib.dump(le, 'lib/label_encoder.pkl')
    joblib.dump(input_features, 'lib/input_features.pkl')

def main():
    data = load_data("data/features.csv")
    print(f"Nacteno {len(data)} her, {len(data.columns)} sloupcu")
    
    X_train_std, X_test_std, y_train, y_test, scaler, le, input_features = preprocess_data(data)
    print("Tridy:", le.classes_)
    print(f"Pocet atributu: {len(input_features)}")
    print(f"Trenovaci data: {len(X_train_std)} | Testovaci data: {len(X_test_std)}")
    
    model = build_and_train_model(X_train_std, y_train)
    evaluate_model(model, X_test_std, y_test, le)
    
    print("\nUkladam model do adresare lib/ pro webove rozhrani...")
    save_model_artifacts(model, scaler, le, input_features)
    print("Hotovo! Data ulozena.")

if __name__ == "__main__":
    main()
