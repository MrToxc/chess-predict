import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from the specified CSV file."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data: pd.DataFrame):
    """
    Splits data into training and testing sets.
    Standardizes the continuous feature data (StandardScaler).
    Target 'was_played' is already binary (0 or 1), so no LabelEncoder is needed.
    """
    
    # We drop 'game_id' so the AI doesn't try to mathematically multiply it
    input_features = [col for col in data.columns if col not in ['was_played', 'game_id']]
    target_feature = 'was_played'

    X_train, X_test, y_train, y_test = train_test_split(
        data[input_features], data[target_feature],
        test_size=0.1, random_state=1
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test, scaler, input_features

def build_and_train_model(X_train, y_train) -> MLPClassifier:
    """
    Creates an MLPClassifier neural network and trains it.
    Since y_train only contains 0 and 1, it will automatically
    train as a Binary Classifier calculating playing probability.
    """
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

def evaluate_model(model: MLPClassifier, X_test, y_test):
    """Prints model metrics on the testing set for Candidate Scoring."""
    y_pred = model.predict(X_test)
    
    print(f"\nMAE:      {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE:      {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Played (0)', 'Played (1)'], zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Example of a few predictions showing the raw probability the AI assigned
    y_proba = model.predict_proba(X_test)

    print(f"\n{'#':<4} {'Prediction (Human Chose It?)':>30} | {'Actual Human Choice':<20}")
    print("-" * 65)
    
    rng = np.random.RandomState(42)
    # Make sure we don't try to sample more rows than exist if the test set is tiny
    sample_size = min(10, len(y_test))
    sample_idx = rng.choice(len(y_test), size=sample_size, replace=False)

    for i in sample_idx:
        proba = y_proba[i]
        # index 1 gives the probability of 'was_played' being 1 (True)
        prob_played = proba[1] * 100
        actual = y_test.iloc[i]
        print(f"{i:<4} {prob_played:>29.1f}% | {actual:<20}")

def save_model_artifacts(model, scaler, input_features):
    os.makedirs('lib', exist_ok=True)
    joblib.dump(model, 'lib/trained_model.pkl')
    joblib.dump(scaler, 'lib/scaler.pkl')
    joblib.dump(input_features, 'lib/input_features.pkl')
    # Because there is no label encoder anymore, we removed that save call.
    # The web interface (app.py) will also need to be updated to no longer
    # require loading a label_encoder.pkl if it was using it.

def main():
    data = load_data("data/features.csv")
    print(f"Loaded {len(data)} candidate move rows, {len(data.columns)} columns")
    
    X_train_std, X_test_std, y_train, y_test, scaler, input_features = preprocess_data(data)
    print(f"Number of attributes (features): {len(input_features)}")
    print(f"Training rows: {len(X_train_std)} | Testing rows: {len(X_test_std)}")
    
    model = build_and_train_model(X_train_std, y_train)
    evaluate_model(model, X_test_std, y_test)
    
    print("\nSaving model to lib/ directory for the web interface...")
    save_model_artifacts(model, scaler, input_features)
    print("Done! Data saved.")

if __name__ == "__main__":
    main()
