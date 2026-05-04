"""
transformer_model.py

Purpose
-------
This script trains and evaluates a transformer-based text classifier.

What this script does
---------------------
1. Loads tokenized text data
2. Uses a pretrained transformer model
3. Fine-tunes the model for multiclass classification
4. Produces predictions on the test set
5. Evaluates the model using shared metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import load_npz
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
 
# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "input_dim":   30000,   # TF-IDF vocabulary size
    "embed_dim":   128,     # Projection dimension
    "num_heads":   4,       # Attention heads
    "num_layers":  2,       # Transformer encoder layers
    "ff_dim":      256,     # Feed-forward hidden size
    "dropout":     0.5,     # Dropout to reduce overfitting
    "num_classes": 14,
    "batch_size":  512,     # Only this many rows are dense at once
    "epochs":      10,
    "lr":          1e-3,
    "weight_decay": 1e-4, 
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
}
 
# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "..", "data", "processed", "tfidf")
MODEL_DIR      = os.path.join(BASE_DIR, "..", "models")
OUT_DIR        = os.path.join(BASE_DIR, "..", "results", "transformer")
MODEL_PATH     = os.path.join(MODEL_DIR, "transformer_model.pt")
 
TRAIN_FEATURES = os.path.join(DATA_DIR, "X_train.npz")
TEST_FEATURES  = os.path.join(DATA_DIR, "X_test.npz")
TRAIN_LABELS   = os.path.join(DATA_DIR, "y_train.npy")
TEST_LABELS    = os.path.join(DATA_DIR, "y_test.npy")
 
 
# ── Sparse Dataset — converts one batch at a time, never the full matrix ───────
class SparseDataset(Dataset):
    """
    Wraps a scipy sparse matrix and a label array.
    Each __getitem__ call converts only a single row to a dense float32
    tensor, so memory usage stays at O(batch_size) not O(n_samples).
    """
    def __init__(self, X_sparse, y):
        self.X = X_sparse   # kept as sparse CSR
        self.y = y
 
    def __len__(self):
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        # Slice one row → dense numpy array → float32 tensor
        x = torch.tensor(
            self.X[idx].toarray().squeeze(0), dtype=torch.float32
        )
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y
 
 
# ── Transformer Model ──────────────────────────────────────────────────────────
class TFIDFTransformer(nn.Module):
    """
    Lightweight Transformer that operates on TF-IDF vectors.
    Projects the sparse input into a dense embedding space,
    prepends a learnable [CLS] token, passes through Transformer
    encoder layers, then classifies via the CLS output.
    """
    def __init__(self, cfg):
        super().__init__()
 
        # Project TF-IDF vector → dense embedding
        self.input_proj = nn.Sequential(
            nn.Linear(cfg["input_dim"], cfg["embed_dim"]),
            nn.LayerNorm(cfg["embed_dim"]),
            nn.Dropout(cfg["dropout"]),
        )
 
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg["embed_dim"]))
 
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["embed_dim"],
            nhead=cfg["num_heads"],
            dim_feedforward=cfg["ff_dim"],
            dropout=cfg["dropout"],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg["num_layers"],
        )
 
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["embed_dim"] // 2),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["embed_dim"] // 2, cfg["num_classes"]),
        )
 
    def forward(self, x):
        x   = self.input_proj(x).unsqueeze(1)        # (batch, 1, embed_dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)             # (batch, 2, embed_dim)
        x   = self.transformer(x)                    # (batch, 2, embed_dim)
        x   = x[:, 0, :]                             # CLS token output
        return self.classifier(x)                    # (batch, num_classes)
 
 
# ── 1. Load Data ───────────────────────────────────────────────────────────────
def load_data():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR,   exist_ok=True)
 
    print("Loading TF-IDF data (sparse — no full dense conversion)...")
    X_train = load_npz(TRAIN_FEATURES)   # stays sparse
    X_test  = load_npz(TEST_FEATURES)    # stays sparse
    y_train = np.load(TRAIN_LABELS, allow_pickle=False)
    y_test  = np.load(TEST_LABELS,  allow_pickle=False)
 
    print(f"  Train: {X_train.shape[0]:,} samples | {X_train.shape[1]:,} features")
    print(f"  Test : {X_test.shape[0]:,} samples  | {X_test.shape[1]:,} features")
    print(f"  Device: {CONFIG['device']}")
 
    # Validate labels
    unique_labels = set(np.unique(y_train).tolist())
    if unique_labels != set(LABEL_MAP.keys()):
        print(f"  WARNING: Labels {unique_labels} do not match LABEL_MAP")
 
    train_loader = DataLoader(
        SparseDataset(X_train, y_train),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,   # keep 0 on Windows to avoid multiprocessing issues
    )
    test_loader = DataLoader(
        SparseDataset(X_test, y_test),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
    )
 
    return train_loader, test_loader, y_test
 
 
# ── 2. Train Model ─────────────────────────────────────────────────────────────
def train_model(train_loader):
    device = CONFIG["device"]
    model  = TFIDFTransformer(CONFIG).to(device)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
 
    print(f"\nTraining Transformer ({CONFIG['epochs']} epochs)...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
 
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item() * y_batch.size(0)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += y_batch.size(0)
 
        scheduler.step()
        print(f"  Epoch {epoch:>2}/{CONFIG['epochs']}  "
              f"Loss: {total_loss/total:.4f}  Acc: {correct/total:.4f}")
 
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")
    return model
 
 
# ── 3. Generate Predictions ────────────────────────────────────────────────────
def predict(model, test_loader):
    print("\nGenerating predictions...")
    device = CONFIG["device"]
    model.eval()
    all_preds = []
 
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            all_preds.extend(model(X_batch).argmax(1).cpu().numpy())
 
    return np.array(all_preds)
 
 
# ── 4. Evaluate Model ──────────────────────────────────────────────────────────
def evaluate(y_test, y_pred):
    print("\n" + "=" * 55)
    print("MODEL EVALUATION — TRANSFORMER")
    print("=" * 55)
 
    avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
    print("=" * 55)
 
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
        pd.DataFrame(report).transpose()
        .drop(columns=["support"], errors="ignore")
    )
    csv_path = os.path.join(OUT_DIR, "classification_report.csv")
    df.to_csv(csv_path)
    print(f"  Saved → {csv_path}")
 
    per_class = df.reindex(labels, fill_value=0)[["precision", "recall", "f1-score"]]
    x, width  = np.arange(len(per_class)), 0.25
 
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
    ax.bar(x - width, per_class["precision"], width, label="Precision", color="#4C72B0")
    ax.bar(x,         per_class["recall"],    width, label="Recall",    color="#DD8452")
    ax.bar(x + width, per_class["f1-score"],  width, label="F1-Score",  color="#55A868")
    ax.set_title("Transformer – Per-Class Metrics", fontsize=13, fontweight="bold")
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
    df_cm  = pd.DataFrame(cm, index=labels, columns=labels)
 
    csv_path = os.path.join(OUT_DIR, "confusion_matrix.csv")
    df_cm.to_csv(csv_path)
    print(f"  Saved → {csv_path}")
 
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.1), max(4, len(labels) * 1.0)))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Count"})
    ax.set_title("Transformer – Confusion Matrix", fontsize=13, fontweight="bold")
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
    train_loader, test_loader, y_test = load_data()
    model                             = train_model(train_loader)
    y_pred                            = predict(model, test_loader)
    evaluate(y_test, y_pred)
