# register_model.py

import os
import argparse
import logging
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# =========================
# ENV + DAGSHUB SETUP
# =========================
load_dotenv()

DAGSHUB_PAT = os.getenv("DAGSHUB_PAT")
if not DAGSHUB_PAT:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_PAT
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_PAT

mlflow.set_tracking_uri(
    "https://dagshub.com/sudarshansahane1044/emo_project.mlflow"
)

# =========================
# LOGGING
# =========================
logger = logging.getLogger("model_registration")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =========================
# PROMOTION LOGIC
# =========================
def promote_model(model_name: str, stage: str, version: int | None):
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    if version is not None:
        selected = next(
            (v for v in versions if int(v.version) == version),
            None
        )
        if not selected:
            raise RuntimeError(
                f"Version {version} not found for model '{model_name}'"
            )
    else:
        selected = max(versions, key=lambda v: int(v.version))

    client.set_model_version_tag(
        name=model_name,
        version=selected.version,
        key="stage",
        value=stage
    )

    logger.info(
        "✅ Model '%s' version %s promoted to '%s'",
        model_name,
        selected.version,
        stage
    )

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Promote MLflow model to a stage"
    )

    parser.add_argument(
        "--model-name",
        default="my_model",
        help="Registered model name"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["staging", "production"],
        help="Target stage"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Specific version to promote (optional)"
    )

    args = parser.parse_args()

    promote_model(
        model_name=args.model_name,
        stage=args.stage,
        version=args.version
    )

if __name__ == "__main__":
    main()
