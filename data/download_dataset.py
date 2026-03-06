"""
download_dataset.py

Purpose
-------
This script downloads the DBPedia-14 dataset using the HuggingFace `datasets` library.

Why we use this script
----------------------
We do NOT store datasets directly in the GitHub repository because:
1. Datasets can be large
2. Git repositories should mainly store code
3. Each team member can generate the dataset locally

What this script does
---------------------
1. Downloads the DBPedia dataset
2. Saves it locally so we don't need to re-download it every run
3. Optionally exports the dataset to CSV for inspection

How to run
----------
python data/download_dataset.py
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

    # load_dataset automatically downloads the dataset from HuggingFace
    dataset = load_dataset(args.dataset_name, cache_dir=str(cache_dir))
    print(dataset)

    print(f"Saving Hugging Face dataset to: {save_disk_dir}")
    
    # save the dataset locally so future runs load from disk instead of downloading again
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
