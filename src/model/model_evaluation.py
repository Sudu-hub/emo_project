import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

model_path = "models/model.pkl"
test_path = "./data/processed/test_bow.csv"
metrics_path = "reports/metrics.json"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test dataset not found at {test_path}")

clf = pickle.load(open(model_path, "rb"))
test_data = pd.read_csv(test_path)

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "auc": auc
}

os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

with open(metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)
