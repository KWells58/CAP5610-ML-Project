"""
naive_bayes.py

Purpose
-------
Train a Multinomial Naive Bayes classifier on the DBPedia-14 dataset
using pre-computed TF-IDF features.

What this script does
---------------------
1. Load pre-processed TF-IDF feature matrices from data/processed/tfidf/
2. Tune hyperparameters using RandomizedSearchCV
3. Train the best model on the full training set
4. Evaluate on the held-out test set
5. Save:
   - Accuracy + classification report (console)
   - Confusion matrix PNG  -> results/plots/
   - Metrics JSON          -> results/
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from scipy.stats import loguniform

# ── Paths ──────────────────────────────────────────────────────────────────────
TFIDF_DIR   = Path("data/processed/tfidf")
RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── DBPedia-14 class names (in label order 0-13) ───────────────────────────────
CLASS_NAMES = [
    "Company",
    "EducationalInstitution",
    "Artist",
    "Athlete",
    "OfficeHolder",
    "MeanOfTransportation",
    "Building",
    "NaturalPlace",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "WrittenWork",
]


# ── 1. Load pre-processed data ─────────────────────────────────────────────────
def load_data() -> tuple:
    print("Loading TF-IDF features...")
    X_train = load_npz(TFIDF_DIR / "X_train.npz")
    X_test  = load_npz(TFIDF_DIR / "X_test.npz")
    y_train = np.load(TFIDF_DIR / "y_train.npy")
    y_test  = np.load(TFIDF_DIR / "y_test.npy")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")
    return X_train, X_test, y_train, y_test


# ── 2. Hyperparameter tuning ───────────────────────────────────────────────────
def tune_model(X_train: object, y_train: np.ndarray) -> MultinomialNB:
    """
    RandomizedSearchCV over:
      - alpha : Laplace / Lidstone smoothing (key NB hyperparameter)
                log-uniform between 0.001 and 10 gives good coverage
      - fit_prior : whether to learn class prior probabilities from data
                    or assume uniform priors
    """
    param_dist = {
        "alpha":     loguniform(1e-3, 10), 
        "fit_prior": [True, False],
    }

    base_model = MultinomialNB()

    search = RandomizedSearchCV(
        estimator           = base_model,
        param_distributions = param_dist,
        n_iter              = 20,         # number of random combinations to try
        cv                  = 5,          # 5-fold cross-validation
        scoring             = "accuracy",
        n_jobs              = -1,         
        random_state        = 42,
        verbose             = 2,
    )

    print("\nRunning RandomizedSearchCV (20 iterations, 5-fold CV)...")
    search.fit(X_train, y_train)

    print(f"\nBest parameters : {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    return search.best_estimator_


# ── 3. Evaluate ────────────────────────────────────────────────────────────────
def evaluate(model: MultinomialNB, X_test: object, y_test: np.ndarray) -> dict:
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report_str)

    return {"accuracy": accuracy, "classification_report": report, "y_pred": y_pred}


# ── 4. Save confusion matrix ───────────────────────────────────────────────────
def save_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = CLASS_NAMES,
        yticklabels = CLASS_NAMES,
        ax          = ax,
        linewidths  = 0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title("Naive Bayes — Confusion Matrix (DBPedia-14)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    out_path = PLOTS_DIR / "naive_bayes_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {out_path}")


# ── 5. Save metrics JSON ───────────────────────────────────────────────────────
def save_metrics(model: MultinomialNB, results: dict) -> None:
    metrics = {
        "model":      "MultinomialNaiveBayes",
        "dataset":    "dbpedia_14",
        "hyperparameters": {
            "alpha":     model.alpha,
            "fit_prior": model.fit_prior,
        },
        "test_accuracy": results["accuracy"],
        "classification_report": results["classification_report"],
    }

    out_path = RESULTS_DIR / "naive_bayes_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to        : {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    best_model                        = tune_model(X_train, y_train)
    results                           = evaluate(best_model, X_test, y_test)
    save_confusion_matrix(y_test, results["y_pred"])
    save_metrics(best_model, results)
    print("\nDone.")


if __name__ == "__main__":
    main()
