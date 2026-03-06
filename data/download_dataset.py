"""
Download and optionally cache the DBPedia-14 dataset locally.

Usage:
    python data/download_dataset.py
    python data/download_dataset.py --save-csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def combine_text(title: str, content: str) -> str:
    title = (title or "").strip()
    content = (content or "").strip()
    if title and content:
        return f"{title}. {content}"
    return title or content


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the DBPedia-14 dataset.")
    parser.add_argument(
        "--dataset-name",
        default="dbpedia_14",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/hf_cache",
        help="Where Hugging Face should cache the dataset.",
    )
    parser.add_argument(
        "--save-disk-dir",
        default="data/raw/dbpedia_14_hf",
        help="Directory where the dataset will be saved with save_to_disk().",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also export train/test CSV files for easy inspection.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    save_disk_dir = Path(args.save_disk_dir)
    save_disk_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, cache_dir=str(cache_dir))
    print(dataset)

    print(f"Saving Hugging Face dataset to: {save_disk_dir}")
    dataset.save_to_disk(str(save_disk_dir))

    if args.save_csv:
        csv_dir = Path("data/raw/csv")
        csv_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "test"):
            split = dataset[split_name]
            out_path = csv_dir / f"{split_name}.csv"

            print(f"Exporting {split_name} split to {out_path}")
            split.to_pandas().assign(
                text=[
                    combine_text(t, c)
                    for t, c in zip(split["title"], split["content"])
                ]
            ).to_csv(out_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
