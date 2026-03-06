"""
Shared tokenizer pipeline for deep learning models.

This script can:
1. Build Hugging Face tokenized datasets for transformer-based models
2. Save NumPy arrays of input_ids, attention_mask, and labels
3. Keep everyone on the same sequence length and tokenizer config

Usage:
    python preprocessing/tokenizer.py
    python preprocessing/tokenizer.py --tokenizer-name distilbert-base-uncased --max-length 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


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


def tokenize_batch(batch, tokenizer, max_length: int):
    texts = [combine_text(t, c) for t, c in zip(batch["title"], batch["content"])]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize DBPedia-14 for deep models.")
    parser.add_argument(
        "--local-path",
        default="data/raw/dbpedia_14_hf",
        help="Optional local save_to_disk() path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/tokenized",
        help="Directory to save tokenized outputs.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="distilbert-base-uncased",
        help="Any Hugging Face tokenizer checkpoint name.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dbpedia(args.local_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"Using tokenizer: {args.tokenizer_name}")
    tokenized = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_length),
        batched=True,
        desc="Tokenizing dataset",
    )

    tokenized.save_to_disk(str(output_dir / "hf_tokenized_dataset"))
    tokenizer.save_pretrained(output_dir / "tokenizer_files")

    for split_name in ("train", "test"):
        split = tokenized[split_name]
        np.save(output_dir / f"{split_name}_input_ids.npy", np.array(split["input_ids"], dtype=np.int32))
        np.save(output_dir / f"{split_name}_attention_mask.npy", np.array(split["attention_mask"], dtype=np.int8))
        np.save(output_dir / f"{split_name}_labels.npy", np.array(split["label"], dtype=np.int64))

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "dbpedia_14",
                "tokenizer_name": args.tokenizer_name,
                "max_length": args.max_length,
            },
            f,
            indent=2,
        )

    print(f"Saved tokenized outputs to: {output_dir}")


if __name__ == "__main__":
    main()
