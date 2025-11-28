#!/usr/bin/env python3
import os
import sys
import logging
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


logger = logging.getLogger("feature_building")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.error("File not found: %s", path)
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.exception("Failed to read CSV at %s: %s", path, e)
        raise
    if df.empty:
        logger.error("Loaded dataframe is empty: %s", path)
        raise ValueError(f"Empty dataframe: {path}")
    return df


def build_bow(
    X_train_texts: pd.Series, X_test_texts: pd.Series, max_features: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train_texts)
        X_test_bow = vectorizer.transform(X_test_texts)
        train_df = pd.DataFrame(X_train_bow.toarray())
        test_df = pd.DataFrame(X_test_bow.toarray())
        return train_df, test_df
    except Exception as e:
        logger.exception("Failed to build BoW features: %s", e)
        raise


def save_dfs(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> None:
    try:
        os.makedirs(out_dir, exist_ok=True)
        train_out = os.path.join(out_dir, "train_bow.csv")
        test_out = os.path.join(out_dir, "test_bow.csv")
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)
        logger.info("Saved train -> %s and test -> %s", train_out, test_out)
    except Exception as e:
        logger.exception("Failed to save CSVs to %s: %s", out_dir, e)
        raise


def main() -> None:
    train_src = "./data/interim/train_processed.csv"
    test_src = "./data/interim/test_processed.csv"
    out_dir = os.path.join("data", "processed")
    try:
        train_data = load_csv(train_src)
        test_data = load_csv(test_src)

        train_data.fillna("", inplace=True)
        test_data.fillna("", inplace=True)

        if "content" not in train_data.columns or "sentiment" not in train_data.columns:
            logger.error("train_processed.csv missing required columns ('content','sentiment')")
            raise KeyError("Missing columns in train_processed.csv")
        if "content" not in test_data.columns or "sentiment" not in test_data.columns:
            logger.error("test_processed.csv missing required columns ('content','sentiment')")
            raise KeyError("Missing columns in test_processed.csv")

        X_train = train_data["content"].astype(str)
        y_train = train_data["sentiment"].values
        X_test = test_data["content"].astype(str)
        y_test = test_data["sentiment"].values

        train_df, test_df = build_bow(X_train, X_test, max_features=50)

        train_df["label"] = y_train
        test_df["label"] = y_test

        save_dfs(train_df, test_df, out_dir)

        logger.info("Feature engineering completed successfully.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Feature engineering failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
