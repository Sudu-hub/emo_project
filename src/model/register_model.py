# register_model.py

import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
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
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_registration_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =========================
# REGISTER / TAG MODEL
# =========================
def register_model(model_name: str, stage: str = "staging"):
    """
    Tag the latest version of a registered model.
    Works with modern MLflow (name= logging).
    """
    try:
        client = MlflowClient()

        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        if not versions:
            raise RuntimeError(f"No versions found for model '{model_name}'")

        # get latest version
        latest_version = max(
            versions,
            key=lambda v: int(v.version)
        )

        client.set_model_version_tag(
            name=model_name,
            version=latest_version.version,
            key="stage",
            value=stage
        )

        logger.info(
            "Model '%s' version %s tagged as '%s'",
            model_name,
            latest_version.version,
            stage
        )

    except Exception as e:
        logger.error("Model registration failed: %s", e)
        raise

# =========================
# MAIN
# =========================
def main():
    model_name = "my_model"
    stage = "staging"   # change to "production" when ready

    register_model(model_name, stage)

if __name__ == "__main__":
    main()
