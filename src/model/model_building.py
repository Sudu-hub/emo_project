import os
import sys
import argparse
import logging
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import yaml

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def load_params(params_path: str):
    if not params_path:
        return {}
    if not os.path.exists(params_path):
        logger.debug("Params file not found at %s — skipping", params_path)
        return {}
    try:
        with open(params_path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as e:
        logger.exception("Failed to load params: %s", e)
        raise


def load_data(train_path: str) -> pd.DataFrame:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at: {train_path}")
    df = pd.read_csv(train_path)
    if df.empty:
        raise ValueError(f"Training dataframe is empty: {train_path}")
    logger.info("Loaded training data %s with shape %s", train_path, df.shape)
    return df


def prepare_X_y(df: pd.DataFrame):
    before = df.shape[0]
    df = df.dropna(how="all")
    dropped = before - df.shape[0]
    if dropped:
        logger.warning("Dropped %d fully-NaN rows", dropped)

    if df.shape[0] == 0:
        raise ValueError("No rows remaining after dropping empty rows.")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    if X.size == 0:
        raise ValueError("No feature columns found.")
    if y.size == 0:
        raise ValueError("Target column is empty.")

    logger.debug("Prepared X shape=%s, y shape=%s", X.shape, y.shape)
    return X, y


def train_model(X, y, n_estimators: int = 50, random_state: int = 42):
    logger.info("Training GradientBoostingClassifier(n_estimators=%s, random_state=%s)", n_estimators, random_state)
    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X, y)
    logger.info("Training complete.")
    return clf


def save_model(model, output_path: str):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s", output_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", type=str, default="./data/processed/train_bow.csv")
    p.add_argument("--model-out", type=str, default="./models/model.pkl")
    p.add_argument("--params", type=str, default="params.yaml")
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--random-state", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        params = load_params(args.params)
        gb_params = params.get("model", {}).get("gradient_boosting", {}) if isinstance(params, dict) else {}

        n_estimators = args.n_estimators if args.n_estimators is not None else int(gb_params.get("n_estimators", 50))
        random_state = args.random_state if args.random_state is not None else int(gb_params.get("random_state", 42))

        df = load_data(args.train_path)
        X, y = prepare_X_y(df)

        model = train_model(X, y, n_estimators=n_estimators, random_state=random_state)
        save_model(model, args.model_out)

        logger.info("Pipeline finished successfully.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

