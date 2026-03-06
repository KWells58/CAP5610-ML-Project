"""
tfidf_pipeline.py

Purpose
-------
This script converts raw text into numerical feature vectors
using TF-IDF (Term Frequency - Inverse Document Frequency).

Why this is needed
------------------
Traditional machine learning models (SVM, Logistic Regression, etc.)
cannot work directly with raw text.

TF-IDF converts text documents into numeric vectors representing
how important words are in each document.

What this script does
---------------------
1. Load the DBPedia dataset
2. Combine title + content fields
3. Build a TF-IDF vocabulary from the training data
4. Transform train and test text into feature vectors
5. Save the resulting matrices for use by ML models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from datasets import load_dataset, load_from_disk
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def combine_text(title: str, content: str) -> str:
    title = (title or "").strip()
    content = (content or "").strip()
    if title and content:
        return f"{title}. {content}"
    return title or content


def load_dbpedia(local_path: str | None) -> dict:
    if local_path and Path(local_path).exists():
        print(f"Loading dataset from disk: {local_path}")
        return load_from_disk(local_path)
    print("Loading dataset from Hugging Face hub...")
    return load_dataset("dbpedia_14")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create TF-IDF features for DBPedia-14.")
    parser.add_argument(
        "--local-path",
        default="data/raw/dbpedia_14_hf",
        help="Optional local save_to_disk() path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/tfidf",
        help="Directory to save processed outputs.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n for n-gram range.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n for n-gram range.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Ignore terms with document frequency lower than this value.",
    )
    parser.add_argument(
        "--sublinear-tf",
        action="store_true",
        help="Apply sublinear tf scaling.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dbpedia(args.local_path)
    train = dataset["train"]
    test = dataset["test"]

    train_texts = [combine_text(t, c) for t, c in zip(train["title"], train["content"])]
    test_texts = [combine_text(t, c) for t, c in zip(test["title"], test["content"])]
    y_train = np.array(train["label"], dtype=np.int64)
    y_test = np.array(test["label"], dtype=np.int64)

    # Create the TF-IDF vectorizer.
    # This converts text into numerical features based on word frequency.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=args.sublinear_tf,
    )

    # Fit the vectorizer on the training data only.
    # This builds the vocabulary and IDF weights.
    print("Fitting TF-IDF vectorizer on training data...")
    X_train = vectorizer.fit_transform(train_texts)

    # Apply the same transformation to the test set.
    # IMPORTANT: we do NOT fit on test data to avoid data leakage.
    print("Transforming test data...")
    X_test = vectorizer.transform(test_texts)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")

    # Save the processed matrices so all team members can load them
    # instead of recomputing TF-IDF every time.
    save_npz(output_dir / "X_train.npz", X_train)
    save_npz(output_dir / "X_test.npz", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "dbpedia_14",
                "max_features": args.max_features,
                "ngram_range": [args.ngram_min, args.ngram_max],
                "min_df": args.min_df,
                "sublinear_tf": args.sublinear_tf,
                "train_shape": list(X_train.shape),
                "test_shape": list(X_test.shape),
            },
            f,
            indent=2,
        )

    print(f"Saved TF-IDF outputs to: {output_dir}")


if __name__ == "__main__":
    main()
