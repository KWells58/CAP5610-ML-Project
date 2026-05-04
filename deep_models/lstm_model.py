"""
lstm_model.py

Purpose
-------
Train a Bidirectional LSTM classifier on the DBPedia-14 dataset
using pre-computed tokenized inputs from tokenizer.py.

Architecture
------------
- Embedding layer (trained from scratch)
- Bidirectional LSTM (2 layers, with dropout)
- Fully connected output layer (14 classes)

What this script does
---------------------
1. Load pre-tokenized inputs from data/processed/tokenized/
2. Subsample 50K training examples (stratified across 14 classes)
3. Split into train / validation sets (90/10)
4. Build and train a BiLSTM with early stopping
5. Evaluate on the held-out test set
6. Save:
   - Accuracy + classification report  -> console
   - Confusion matrix PNG              -> results/plots/
   - Metrics JSON                      -> results/
   - Trained model weights             -> results/lstm_model.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
TOKENIZED_DIR = Path("data/processed/tokenized")
RESULTS_DIR   = Path("results")
PLOTS_DIR     = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── DBPedia-14 class names ─────────────────────────────────────────────────────
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
NUM_CLASSES = len(CLASS_NAMES)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SUBSAMPLE_SIZE  = 50_000   # training examples to use
VAL_SPLIT       = 0.10     # 10% of subsample used for validation
EMBED_DIM       = 128      # embedding vector size
HIDDEN_DIM      = 256      # LSTM hidden state size
NUM_LAYERS      = 2        # stacked BiLSTM layers
DROPOUT         = 0.3      # dropout between LSTM layers
BATCH_SIZE      = 256
LEARNING_RATE   = 1e-3
MAX_EPOCHS      = 20
PATIENCE        = 3        # early stopping: stop after N epochs of no improvement


# ── 1. Dataset ─────────────────────────────────────────────────────────────────
class DBPediaDataset(Dataset):
    """Wraps tokenized input_ids and labels as a PyTorch Dataset."""

    def __init__(self, input_ids: np.ndarray, labels: np.ndarray) -> None:
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.labels[idx]


def stratified_subsample(
    input_ids: np.ndarray,
    labels: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample exactly n examples from input_ids/labels, keeping class balance.
    Each of the 14 classes gets n // NUM_CLASSES examples.
    """
    per_class = n // NUM_CLASSES
    indices = []
    for cls in range(NUM_CLASSES):
        cls_idx = np.where(labels == cls)[0]
        chosen  = np.random.choice(cls_idx, size=per_class, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    np.random.shuffle(indices)
    return input_ids[indices], labels[indices]


def load_data(full_dataset: bool = False) -> tuple:
    print("Loading tokenized data...")
    train_ids    = np.load(TOKENIZED_DIR / "train_input_ids.npy")
    train_labels = np.load(TOKENIZED_DIR / "train_labels.npy")
    test_ids     = np.load(TOKENIZED_DIR / "test_input_ids.npy")
    test_labels  = np.load(TOKENIZED_DIR / "test_labels.npy")

    print(f"  Full train: {train_ids.shape}  test: {test_ids.shape}")

    if full_dataset:
        print("  Using full training dataset (no subsampling).")
    else:
        # Subsample training data (stratified)
        train_ids, train_labels = stratified_subsample(train_ids, train_labels, SUBSAMPLE_SIZE)
        print(f"  Subsampled train: {train_ids.shape}")

    # Infer vocab size from the tokenized data (DistilBERT vocab = 30522)
    vocab_size = int(train_ids.max()) + 1
    seq_len    = train_ids.shape[1]
    print(f"  Vocab size: {vocab_size}  Sequence length: {seq_len}")

    return train_ids, train_labels, test_ids, test_labels, vocab_size, seq_len


# ── 2. Model ───────────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM text classifier.

    Architecture:
        Embedding → BiLSTM (2 layers) → take final hidden state → Linear → 14 classes

    The bidirectional design reads each sequence both left-to-right and
    right-to-left, then concatenates both final hidden states before
    passing to the classifier head. This captures context from both
    directions, improving accuracy over a unidirectional LSTM.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
            padding_idx    = pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,        # read left→right AND right→left
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # BiLSTM outputs hidden_dim * 2 (forward + backward)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))          
        _, (hidden, _) = self.lstm(embedded)          

        # Grab the final layer's forward and backward hidden states
        # hidden[-2] = last forward layer, hidden[-1] = last backward layer
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  
        out = self.fc(self.dropout(final_hidden))       
        return out


# ── 3. Training ────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Train with early stopping.
    Saves the best model weights (lowest validation loss) to disk.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss   = float("inf")
    epochs_no_improve = 0
    history         = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_model_path = RESULTS_DIR / "lstm_model.pt"

    print(f"\nTraining BiLSTM for up to {MAX_EPOCHS} epochs (early stopping patience={PATIENCE})...")
    print("-" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        t0         = time.time()
        train_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits      = model(input_ids)
                loss        = criterion(logits, labels)
                val_loss   += loss.item()
                preds       = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= len(val_loader)
        val_acc   = val_correct / val_total
        elapsed   = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    print(f"\nBest model weights saved to: {best_model_path}")
    return history


# ── 4. Evaluate ────────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            logits    = model(input_ids)
            preds     = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy   = accuracy_score(all_labels, all_preds)
    report_str = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    report     = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report_str)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "y_pred": all_preds,
        "y_true": all_labels,
    }


# ── 5. Save confusion matrix ───────────────────────────────────────────────────
def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Purples",
        xticklabels = CLASS_NAMES,
        yticklabels = CLASS_NAMES,
        ax          = ax,
        linewidths  = 0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title("BiLSTM — Confusion Matrix (DBPedia-14)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    out_path = PLOTS_DIR / "lstm_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {out_path}")


# ── 6. Save metrics ────────────────────────────────────────────────────────────
def save_metrics(results: dict, history: dict, full_dataset: bool = False) -> None:
    metrics = {
        "model":   "BiLSTM",
        "dataset": "dbpedia_14",
        "hyperparameters": {
            "training_set":   "full (560K)" if full_dataset else f"subsampled ({SUBSAMPLE_SIZE:,})",
            "embed_dim":      EMBED_DIM,
            "hidden_dim":     HIDDEN_DIM,
            "num_layers":     NUM_LAYERS,
            "dropout":        DROPOUT,
            "batch_size":     BATCH_SIZE,
            "learning_rate":  LEARNING_RATE,
            "max_epochs":     MAX_EPOCHS,
            "patience":       PATIENCE,
        },
        "test_accuracy":          results["accuracy"],
        "classification_report":  results["classification_report"],
        "training_history": {
            "train_loss": history["train_loss"],
            "val_loss":   history["val_loss"],
            "val_acc":    history["val_acc"],
        },
    }

    out_path = RESULTS_DIR / "lstm_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BiLSTM on DBPedia-14.")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Train on the full 560K dataset instead of the default 50K subsample.",
    )
    args = parser.parse_args()

    # Device: use MPS (Apple Silicon GPU) if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    if args.full_dataset:
        print("Mode: FULL dataset (560K training samples)")
    else:
        print(f"Mode: Subsampled ({SUBSAMPLE_SIZE:,} training samples) — pass --full-dataset to use all data")

    # Load data
    train_ids, train_labels, test_ids, test_labels, vocab_size, seq_len = load_data(args.full_dataset)

    # Build datasets
    full_train_dataset = DBPediaDataset(train_ids, train_labels)
    test_dataset       = DBPediaDataset(test_ids,  test_labels)

    # Train / validation split (90/10)
    val_size   = int(len(full_train_dataset) * VAL_SPLIT)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"\n  Train samples : {train_size:,}")
    print(f"  Val samples   : {val_size:,}")
    print(f"  Test samples  : {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = BiLSTMClassifier(
        vocab_size  = vocab_size,
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_layers  = NUM_LAYERS,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # Train
    history = train_model(model, train_loader, val_loader, device)

    # Load best weights before evaluating
    model.load_state_dict(torch.load(RESULTS_DIR / "lstm_model.pt", map_location=device))

    # Evaluate
    results = evaluate(model, test_loader, device)

    # Save outputs
    save_confusion_matrix(results["y_true"], results["y_pred"])
    save_metrics(results, history, args.full_dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
