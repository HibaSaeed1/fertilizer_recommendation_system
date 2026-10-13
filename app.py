"""
app.py — Flask backend for the Fertilizer Recommendation System.

Loads the trained model + label encoders (from train_model.py) and serves:
  - GET  /          -> the frontend form
  - POST /predict   -> takes soil/crop/environment readings, returns the
                        recommended fertilizer
  - GET  /options    -> valid Soil Type / Crop Type choices, for the dropdowns
"""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "fertilizer_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")

model = None
encoders = None
load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
except Exception as e:  # noqa: BLE001
    load_error = str(e)

NUMERIC_FIELDS = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/options")
def options():
    if encoders is None:
        return jsonify({"error": load_error}), 500
    return jsonify({
        "soil_types": encoders["soil_encoder"].classes_.tolist(),
        "crop_types": encoders["crop_encoder"].classes_.tolist(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or encoders is None:
        return jsonify({"error": f"Model not loaded: {load_error}"}), 500

    data = request.get_json(silent=True) or request.form.to_dict()

    missing = [f for f in NUMERIC_FIELDS + ["Soil Type", "Crop Type"] if f not in data or data[f] in (None, "")]
    if missing:
        return jsonify({"error": f"Missing field(s): {', '.join(missing)}"}), 400

    try:
        numeric_values = {f: float(data[f]) for f in NUMERIC_FIELDS}
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"All numeric fields must be numbers. ({e})"}), 400

    soil_encoder = encoders["soil_encoder"]
    crop_encoder = encoders["crop_encoder"]
    label_encoder = encoders["label_encoder"]
    feature_cols = encoders["feature_cols"]

    if data["Soil Type"] not in soil_encoder.classes_:
        return jsonify({"error": f"Unknown Soil Type '{data['Soil Type']}'. Valid: {list(soil_encoder.classes_)}"}), 400
    if data["Crop Type"] not in crop_encoder.classes_:
        return jsonify({"error": f"Unknown Crop Type '{data['Crop Type']}'. Valid: {list(crop_encoder.classes_)}"}), 400

    row = {**numeric_values}
    row["Soil Type_enc"] = int(soil_encoder.transform([data["Soil Type"]])[0])
    row["Crop Type_enc"] = int(crop_encoder.transform([data["Crop Type"]])[0])

    input_df = pd.DataFrame([row], columns=feature_cols)

    pred_encoded = model.predict(input_df)[0]
    fertilizer = label_encoder.inverse_transform([pred_encoded])[0]

    top3 = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        ranked = sorted(zip(label_encoder.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
        top3 = [{"fertilizer": f, "confidence": round(float(p) * 100, 2)} for f, p in ranked]

    return jsonify({
        "recommended_fertilizer": fertilizer,
        "top3": top3,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
