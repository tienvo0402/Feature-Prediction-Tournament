# Feature Prediction Tournament Client (1st Place)

UTSA CS 6243 Machine Learning Project 1 — placed **1st** in the class feature-prediction tournament.

## What it does
- Connects a Python client to the course prediction server
- Trains on labeled feature data (`StandardScaler` + `GridSearchCV`-tuned **k-NN**)
- Predicts holdout class labels from features only
- Includes generated multi-class datasets used to challenge the course dataset generator / predictors

## Requirements
```bash
pip install scikit-learn numpy matplotlib
```

## Run (local demo)
```bash
# Terminal 1
python server.py --rounds 10

# Terminal 2 (provides dataset)
python client.py --dataset moderate_5class_dataset.csv

# Terminal 3 (second client / opponent slot)
python client.py
```

## Files
- `client.py` — trained predictor client
- `server.py` — course-style server (train/holdout split + inference rounds)
- `example.csv`, `moderate_2feature_dataset.csv`, `moderate_5class_dataset.csv` — sample / generated datasets
