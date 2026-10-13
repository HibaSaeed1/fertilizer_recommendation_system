# Fertilizer Recommendation System

A Machine Learning system that recommends the most suitable fertilizer for a field based on soil nutrients (N, P, K), soil type, crop type, and environmental readings (temperature, humidity, moisture).

## Project structure

```
.
├── notebooks/
│   └── Fertilizer_Recommendation_ML.ipynb   # Full ML workflow: EDA, model comparison, evaluation
├── data/
│   └── raw/
│       └── Fertilizer_Prediction.csv        # Dataset — 99 samples, 7 fertilizer classes
├── models/
│   ├── fertilizer_model.pkl                 # Trained model (Random Forest)
│   ├── encoders.pkl                         # Label encoders for Soil Type / Crop Type / target
│   └── metrics.json                         # Saved evaluation metrics
├── train_model.py                           # Training pipeline (run to regenerate the model)
├── app.py                                   # Flask backend
├── templates/
│   └── index.html                           # Frontend form
├── static/
│   ├── style.css
│   └── script.js
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```
This preprocesses the data, compares Decision Tree / Random Forest / KNN, picks the best by F1-score, evaluates it (accuracy, precision, recall, F1, confusion matrix), and saves `models/fertilizer_model.pkl` + `models/encoders.pkl`.

## Run the app

```bash
python app.py
```
Open **http://localhost:5000**. Fill in the environment readings, soil nutrients, soil type, and crop type, and submit for a fertilizer recommendation with the next two closest matches.

## Explore the notebook

```bash
cd notebooks
jupyter notebook
```
Covers the full workflow required by the task: load → clean → EDA (distribution plots, correlation heatmap, boxplots) → encode → split → train & compare 3 models → evaluate → predict.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend form |
| `/options` | GET | Valid Soil Type / Crop Type values, for populating dropdowns |
| `/predict` | POST | Takes `Temparature`, `Humidity`, `Moisture`, `Nitrogen`, `Potassium`, `Phosphorous`, `Soil Type`, `Crop Type`; returns the recommended fertilizer + top-3 confidences |
| `/health` | GET | `{"status": "ok"}` |

Example:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Temparature":"26","Humidity":"52","Moisture":"38","Nitrogen":"37","Potassium":"0","Phosphorous":"0","Soil Type":"Sandy","Crop Type":"Maize"}'
```

## Model performance

Random Forest was automatically selected as the best of 3 candidates (Decision Tree, Random Forest, KNN) by weighted F1-score. See `models/metrics.json` for the full comparison and confusion matrix.

## Dataset

Source: the widely-used Kaggle "Fertilizer Prediction" dataset (99 rows, 7 fertilizer classes: Urea, DAP, 14-35-14, 17-17-17, 20-20, 28-28, 10-26-26; 5 soil types; 11 crop types).

## Tech stack

- **Modeling**: Python, pandas, scikit-learn, seaborn/matplotlib (EDA)
- **Backend**: Flask
- **Frontend**: HTML, CSS, vanilla JavaScript
Testing git diff command
