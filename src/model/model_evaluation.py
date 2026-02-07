import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

# =========================
# ENV + DAGSHUB SETUP
# =========================
load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "sudarshansahane1044"
repo_name = "emo_project"

mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

# =========================
# LOGGING
# =========================
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_evaluation_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =========================
# HELPERS
# =========================
def load_model(file_path: str):
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded from %s", file_path)
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise

def evaluate_model(clf, X_test, y_test) -> dict:
    try:
        y_pred = clf.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
            metrics["auc"] = roc_auc_score(y_test, y_prob)

        logger.debug("Evaluation metrics computed")
        return metrics

    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise

def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.debug("Saved file: %s", path)

# =========================
# MAIN
# =========================
def main():
    mlflow.set_experiment("dvcpipes")

    with mlflow.start_run() as run:
        try:
            clf = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_bow.csv")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)

            os.makedirs("reports", exist_ok=True)
            save_json(metrics, "reports/metrics.json")

            # --- log metrics & params ---
            mlflow.log_metrics(metrics)
            if hasattr(clf, "get_params"):
                mlflow.log_params(clf.get_params())

            # ✅ MODERN MLflow: auto-register model
            mlflow.sklearn.log_model(
                sk_model=clf,
                name="my_model",          # 👈 registered model name
                input_example=X_test[:5] # 👈 signature inference
            )

            # --- log artifacts ---
            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("model_evaluation_errors.log")

            logger.info("Run completed successfully: %s", run.info.run_id)

        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            raise

if __name__ == "__main__":
    main()
