import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from typing import Optional

# Initialize DagsHub + MLflow
dagshub.init(
    repo_owner='sudarshansahane1044',
    repo_name='emo_project',
    mlflow=True
)
# dagshub.init sets the MLflow tracking URI internally

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def log_if_exists(path: str, artifact_subdir: Optional[str] = None) -> None:
    """
    Log a file or directory to MLflow if it exists.
    - If 'path' is a file  -> mlflow.log_artifact
    - If 'path' is a dir   -> mlflow.log_artifacts
    """
    if not os.path.exists(path):
        logger.warning("Artifact path does not exist, skipping: %s", path)
        return

    try:
        if os.path.isdir(path):
            mlflow.log_artifacts(path, artifact_path=artifact_subdir)
        else:
            mlflow.log_artifact(path, artifact_path=artifact_subdir)
        logger.debug("Logged artifact: %s (subdir: %s)", path, artifact_subdir)
    except Exception as e:
        logger.error("Failed to log artifact %s: %s", path, e)
        # Do not raise so the rest of the pipeline can continue


def main():
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)

    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            # 1. Load model and data
            model_path_local = './models/model.pkl'
            clf = load_model(model_path_local)
            test_data = load_data('./data/processed/test_bow.csv')

            # 2. Split features and target
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # 3. Evaluate
            metrics = evaluate_model(clf, X_test, y_test)

            metrics_file = 'reports/metrics.json'
            model_info_file = 'reports/model_info.json'

            # 4. Save metrics locally
            save_metrics(metrics, metrics_file)

            # 5. Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # 6. Log model hyperparameters to MLflow (if sklearn-like estimator)
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # 7. Save full MLflow model locally (MLmodel, model.pkl, env files, etc.)
            local_mlflow_model_dir = "mlflow_model"
            mlflow.sklearn.save_model(
                sk_model=clf,
                path=local_mlflow_model_dir
            )

            # 8. Log the entire model directory as artifacts to MLflow/DagsHub
            mlflow.log_artifacts(
                local_mlflow_model_dir,
                artifact_path="model"
            )
            # In artifacts you will see: model/MLmodel, model/model.pkl, model/conda.yaml, etc.

            # 9. Save model info (run_id + folder where MLflow-model is stored)
            save_model_info(run.info.run_id, "model", model_info_file)

            # 10. Log important local artifacts
            mlflow.log_artifact(metrics_file)
            mlflow.log_artifact(model_info_file)
            mlflow.log_artifact('model_evaluation_errors.log')

            # 11. Optionally log code, configs, and environment files
            log_if_exists("src", artifact_subdir="code")
            log_if_exists("dvc.yaml")
            log_if_exists("params.yaml")
            log_if_exists("config.yaml")
            log_if_exists("requirements.txt")
            log_if_exists("environment.yml")

        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()

