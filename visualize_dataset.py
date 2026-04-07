"""
visualize_dataset.py

Purpose
-------
This script performs basic exploratory data analysis (EDA) on the DBPedia-14 dataset.

Visualizations produced:
1. Class distribution
2. Document length distribution
3. Top TF-IDF terms

How to run
----------
python visualize_dataset.py
"""

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


def combine_text(title, content):
    """Combine title and content into one string."""
    return f"{title}. {content}"


def main():

    print("Loading dataset...")
    dataset = load_dataset("dbpedia_14")

    train = dataset["train"]

    # Combine title + content
    texts = [combine_text(t, c) for t, c in zip(train["title"], train["content"])]
    labels = train["label"]

    print("Dataset size:", len(texts))

    # -------------------------------
    # Class distribution
    # -------------------------------

    counts = Counter(labels)

    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.title("Class Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")

    plt.show()

    # -------------------------------
    # Document length distribution
    # -------------------------------

    lengths = [len(text.split()) for text in texts[:20000]]

    plt.figure()
    plt.hist(lengths, bins=50)
    plt.title("Document Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    plt.show()

    # -------------------------------
    # Top TF-IDF Terms
    # -------------------------------

    print("Computing TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X = vectorizer.fit_transform(texts[:20000])

    tfidf_scores = np.mean(X.toarray(), axis=0)

    terms = vectorizer.get_feature_names_out()

    top_indices = np.argsort(tfidf_scores)[-20:]

    top_terms = [terms[i] for i in top_indices]
    top_scores = tfidf_scores[top_indices]

    plt.figure()

    plt.barh(top_terms, top_scores)

    plt.title("Top TF-IDF Terms")

    plt.xlabel("Average TF-IDF Score")

    plt.show()


if __name__ == "__main__":
    main()