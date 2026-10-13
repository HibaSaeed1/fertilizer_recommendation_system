"""
train_model.py — Fertilizer Recommendation System: training pipeline.

Covers the required steps: preprocessing (encode categoricals, no missing/
duplicate values here but checked anyway), EDA is in the notebook, model
comparison (Decision Tree, Random Forest, KNN), evaluation (accuracy,
precision, recall, F1, confusion matrix), and saving the final model +
encoders for the app to use.

Run:
    python train_model.py
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_PATH = "data/raw/Fertilizer_Prediction.csv"
MODEL_PATH = "models/fertilizer_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
METRICS_PATH = "models/metrics.json"

NUMERIC_FEATURES = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL_FEATURES = ["Soil Type", "Crop Type"]
TARGET = "Fertilizer Name"


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    # --- Preprocessing ---
    print(f"Missing values:\n{df.isnull().sum().sum()} total")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    df = df.drop_duplicates()

    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    label_encoder = LabelEncoder()

    df["Soil Type_enc"] = soil_encoder.fit_transform(df["Soil Type"])
    df["Crop Type_enc"] = crop_encoder.fit_transform(df["Crop Type"])
    y = label_encoder.fit_transform(df[TARGET])

    feature_cols = NUMERIC_FEATURES + ["Soil Type_enc", "Crop Type_enc"]
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Model comparison ---
    candidates = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
    }

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
        }
        print(f"\n{name}: {results[name]}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = candidates[best_name]
    print(f"\nBest model: {best_name}")

    # --- Final evaluation on the chosen model ---
    final_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, final_preds).tolist()
    report = classification_report(
        y_test, final_preds, target_names=label_encoder.classes_, zero_division=0
    )
    print("\nConfusion matrix:\n", np.array(cm))
    print("\nClassification report:\n", report)

    # --- Save model, encoders, and metrics ---
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump({
            "soil_encoder": soil_encoder,
            "crop_encoder": crop_encoder,
            "label_encoder": label_encoder,
            "feature_cols": feature_cols,
        }, f)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "best_model": best_name,
            "all_results": results,
            "confusion_matrix": cm,
            "classes": label_encoder.classes_.tolist(),
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}, encoders to {ENCODERS_PATH}, metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
