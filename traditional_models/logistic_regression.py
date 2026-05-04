"""
logistic_regression.py

Purpose
-------
This script trains and evaluates a Logistic Regression classifier
using the shared TF-IDF features.

What this script does
---------------------
1. Loads the processed TF-IDF training and test data
2. Trains a Logistic Regression model
3. Generates predictions on the test set
4. Evaluates the model using shared metrics
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
 
# ── Label Mapping ──────────────────────────────────────────────────────────────
LABEL_MAP = {
    0:  "Company",
    1:  "EducationalInstitution",
    2:  "Artist",
    3:  "Athlete",
    4:  "OfficeHolder",
    5:  "MeanOfTransportation",
    6:  "Building",
    7:  "NaturalPlace",
    8:  "Village",
    9:  "Animal",
    10: "Plant",
    11: "Album",
    12: "Film",
    13: "WrittenWork",
}
 
# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "processed", "tfidf")
MODEL_DIR = os.path.join("models")
OUT_DIR   = os.path.join("results", "logistic_regression")
 
TRAIN_FEATURES = os.path.join(DATA_DIR, "X_train.npz")
TEST_FEATURES  = os.path.join(DATA_DIR, "X_test.npz")
TRAIN_LABELS   = os.path.join(DATA_DIR, "y_train.npy")
TEST_LABELS    = os.path.join(DATA_DIR, "y_test.npy")
MODEL_PATH     = os.path.join(MODEL_DIR, "logistic_regression.joblib")
 
 
# ── 1. Load Data ───────────────────────────────────────────────────────────────
def load_data():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR,   exist_ok=True)
 
    print("Loading TF-IDF data...")
    X_train = load_npz(TRAIN_FEATURES)
    X_test  = load_npz(TEST_FEATURES)
    y_train = np.load(TRAIN_LABELS, allow_pickle=False)
    y_test  = np.load(TEST_LABELS,  allow_pickle=False)
 
    print(f"  Train: {X_train.shape[0]:,} samples | {X_train.shape[1]:,} features")
    print(f"  Test : {X_test.shape[0]:,} samples  | {X_test.shape[1]:,} features")
 
    # Validate label IDs match LABEL_MAP
    unique_labels = set(np.unique(y_train).tolist())
    expected      = set(LABEL_MAP.keys())
    if unique_labels != expected:
        print(f"  WARNING: Labels in data {unique_labels} do not match LABEL_MAP {expected}")
 
    return X_train, X_test, y_train, y_test
 
 
# ── 2. Train Model ─────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        C=0.1,           # Regularization to reduce overfitting
        max_iter=1000,
        solver="saga",   # saga supports n_jobs and large datasets
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
 
    # Convergence check
    iters = model.n_iter_[0] if hasattr(model.n_iter_, "__len__") else model.n_iter_
    if iters >= model.max_iter:
        print("  WARNING: Model did not converge — consider increasing max_iter")
 
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")
    return model
 
 
# ── 3. Generate Predictions ────────────────────────────────────────────────────
def predict(model, X_test):
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    return y_pred
 
 
# ── 4. Evaluate Model ──────────────────────────────────────────────────────────
def evaluate(y_test, y_pred):
    print("\n" + "=" * 55)
    print("MODEL EVALUATION")
    print("=" * 55)
 
    avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
 
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
    print("=" * 55)
 
    # Convert numeric → word labels for all outputs
    y_test_named = np.array([LABEL_MAP.get(int(l), str(l)) for l in y_test])
    y_pred_named = np.array([LABEL_MAP.get(int(l), str(l)) for l in y_pred])
 
    save_tables(y_test_named, y_pred_named)
 
 
# ── 5. Save Classification Report ─────────────────────────────────────────────
def save_classification_report(y_test, y_pred):
    labels = [LABEL_MAP[k] for k in sorted(LABEL_MAP.keys())]
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
 
    df = (
        pd.DataFrame(report)
        .transpose()
        .drop(columns=["support"], errors="ignore")
    )
    csv_path = os.path.join(OUT_DIR, "classification_report.csv")
    df.to_csv(csv_path, index=True)
    print(f"  Saved → {csv_path}")
 
    per_class = df.reindex(labels, fill_value=0)[["precision", "recall", "f1-score"]]
    x     = np.arange(len(per_class))
    width = 0.25
 
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
    ax.bar(x - width, per_class["precision"], width, label="Precision", color="#4C72B0")
    ax.bar(x,         per_class["recall"],    width, label="Recall",    color="#DD8452")
    ax.bar(x + width, per_class["f1-score"],  width, label="F1-Score",  color="#55A868")
 
    ax.set_title("Logistic Regression – Per-Class Metrics", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class Label", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(per_class.index, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
 
    png_path = os.path.join(OUT_DIR, "classification_report.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {png_path}")
 
 
# ── 6. Save Confusion Matrix ───────────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred):
    labels = [LABEL_MAP[k] for k in sorted(LABEL_MAP.keys())]
    cm     = confusion_matrix(y_test, y_pred, labels=labels)
 
    df_cm    = pd.DataFrame(cm, index=labels, columns=labels)
    csv_path = os.path.join(OUT_DIR, "confusion_matrix.csv")
    df_cm.to_csv(csv_path)
    print(f"  Saved → {csv_path}")
 
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.1), max(4, len(labels) * 1.0)))
    sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title("Logistic Regression – Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
 
    png_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {png_path}")
 
 
# ── 7. Save All Tables & Plots ─────────────────────────────────────────────────
def save_tables(y_test, y_pred):
    print(f"\nSaving tables & plots to {OUT_DIR}/")
    save_classification_report(y_test, y_pred)
    save_confusion_matrix(y_test, y_pred)
    print("Done.")
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model                            = train_model(X_train, y_train)
    y_pred                           = predict(model, X_test)
    evaluate(y_test, y_pred)
