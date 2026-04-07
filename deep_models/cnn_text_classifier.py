"""
cnn_text_classifier.py

Train and evaluate a simple 1D CNN for text classification.

Run:
    python -m deep_models.cnn_text_classifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import evaluate_classification, print_metrics


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_filters: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)              # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)            # (batch_size, embed_dim, seq_len)
        x = self.conv(x)                  # (batch_size, num_filters, new_seq_len)
        x = self.relu(x)
        x = torch.max(x, dim=2).values    # global max pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a 1D CNN for text classification.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/tokenized",
        help="Directory containing tokenized numpy arrays",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=128,
        help="Number of convolution filters",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Conv1D kernel size",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=50000,
        help="Optional cap on training examples for faster experimentation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: cuda or cpu",
    )
    return parser.parse_args()


def load_data(data_dir: str, train_limit: int | None):
    data_path = Path(data_dir)

    X_train = np.load(data_path / "train_input_ids.npy")
    X_test = np.load(data_path / "test_input_ids.npy")
    y_train = np.load(data_path / "train_labels.npy")
    y_test = np.load(data_path / "test_labels.npy")

    if train_limit is not None and train_limit < len(X_train):
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]

    return X_train, X_test, y_train, y_test


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    X_train, X_test, y_train, y_test = load_data(args.data_dir, args.train_limit)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Using device: {device}")

    # vocab size estimate from tokenizer ids
    vocab_size = int(max(X_train.max(), X_test.max())) + 1
    num_classes = len(np.unique(y_train))

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Training CNN...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    print("Generating predictions...")
    y_pred = predict(model, test_loader, device)

    metrics = evaluate_classification(y_test, y_pred)

    print("\nCNN Results")
    print("-" * 40)
    print_metrics(metrics)

    print("\nAdditional Model Info")
    print("-" * 40)
    print(f"Vocab size    : {vocab_size}")
    print(f"Num classes   : {num_classes}")
    print(f"Embedding dim : {args.embed_dim}")
    print(f"Num filters   : {args.num_filters}")
    print(f"Kernel size   : {args.kernel_size}")
    print(f"Epochs        : {args.epochs}")


if __name__ == "__main__":
    main()