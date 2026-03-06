"""
dataset_stats.py

Purpose
-------
This script computes basic statistics for the DBPedia dataset.

Why this matters
----------------
Understanding the dataset helps us verify:
- the number of samples
- class distribution
- average document length

These statistics are useful for:
- validating dataset integrity
- writing the project report
- understanding model performance later

How to run
----------
python dataset_stats.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk


LABEL_NAMES = [
    "Company",
    "Educational Institution",
    "Artist",
    "Athlete",
    "Office Holder",
    "Mean Of Transportation",
    "Building",
    "Natural Place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written Work",
]


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
    parser = argparse.ArgumentParser(description="Compute DBPedia-14 dataset stats.")
    parser.add_argument(
        "--local-path",
        default="data/raw/dbpedia_14_hf",
        help="Optional local save_to_disk() path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="How many training examples to sample for average length stats.",
    )
    args = parser.parse_args()

    dataset = load_dbpedia(args.local_path)
    train = dataset["train"]
    test = dataset["test"]

    y_train = train["label"]
    y_test = test["label"]

    # Combine the title and content fields into a single text input.
    # Most text classification models expect one input string per sample.
    sampled_texts = [
        combine_text(train[i]["title"], train[i]["content"])
        for i in range(min(args.sample_size, len(train)))
    ]
    token_lengths = [len(text.split()) for text in sampled_texts]
    char_lengths = [len(text) for text in sampled_texts]

    # We only sample part of the dataset to compute statistics
    # because computing over all 560k samples would be slower
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    print("=" * 60)
    print("DBPEDIA-14 DATASET STATISTICS")
    print("=" * 60)
    print(f"Train samples: {len(train):,}")
    print(f"Test samples : {len(test):,}")
    print(f"Total samples: {len(train) + len(test):,}")
    print(f"Number of classes: {len(set(y_train))}")
    print()
    print(f"Average document length (words, sample): {np.mean(token_lengths):.2f}")
    print(f"Median document length (words, sample) : {np.median(token_lengths):.2f}")
    print(f"Average document length (chars, sample): {np.mean(char_lengths):.2f}")
    print()

    print("Class distribution (train):")
    for label_idx in sorted(train_counts):
        label_name = LABEL_NAMES[label_idx] if label_idx < len(LABEL_NAMES) else str(label_idx)
        print(f"  {label_idx:>2} - {label_name:<25}: {train_counts[label_idx]:>7,}")

    print()
    print("Class distribution (test):")
    for label_idx in sorted(test_counts):
        label_name = LABEL_NAMES[label_idx] if label_idx < len(LABEL_NAMES) else str(label_idx)
        print(f"  {label_idx:>2} - {label_name:<25}: {test_counts[label_idx]:>7,}")

    print("=" * 60)


if __name__ == "__main__":
    main()
