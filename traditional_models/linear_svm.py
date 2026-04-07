"""
linear_svm.py

Train and evaluate a Linear SVM on the shared TF-IDF features.

Run:
    python -m traditional_models.linear_svm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC

from evaluation.metrics import evaluate_classification, print_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a Linear SVM.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/tfidf",
        help="Directory containing X_train.npz, X_test.npz, y_train.npy, y_test.npy",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization strength",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_data(data_dir: str):
    data_path = Path(data_dir)
    X_train = load_npz(data_path / "X_train.npz")
    X_test = load_npz(data_path / "X_test.npz")
    y_train = np.load(data_path / "y_train.npy")
    y_test = np.load(data_path / "y_test.npy")
    return X_train, X_test, y_train, y_test


def main() -> None:
    args = parse_args()

    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")

    model = LinearSVC(
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    print("Training Linear SVM...")
    model.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    metrics = evaluate_classification(y_test, y_pred)

    print("\nLinear SVM Results")
    print("-" * 40)
    print_metrics(metrics)

    print("\nAdditional Model Info")
    print("-" * 40)
    print(f"Number of input features: {X_train.shape[1]}")
    print(f"C value                : {args.C}")
    print(f"Max iterations         : {args.max_iter}")


if __name__ == "__main__":
    main()