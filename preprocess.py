#!/usr/bin/env python
"""
Dataset preprocessing for Go-Browse training pipeline.
Loads apurvaga/go-browse-wa, cleans, converts to training format, creates splits.

Usage:
    python preprocess.py                          # download from HF + process
    python preprocess.py --local data_local/      # use local copy
    python preprocess.py --full                   # process full dataset (not just demo)
    python preprocess.py --val-size 0.05           # custom validation split ratio
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def inspect_dataset(ds, name="dataset"):
    """Print dataset schema and stats."""
    logger.info(f"--- {name} ---")
    logger.info(f"  Num samples: {len(ds)}")
    logger.info(f"  Features: {ds.features}")

    sample = ds[0]
    logger.info(f"  Sample keys: {list(sample.keys())}")

    # Stats on traj_reward
    if "traj_reward" in ds.features:
        rewards = [x["traj_reward"] for x in ds]
        pos = sum(1 for r in rewards if r > 0)
        neg = sum(1 for r in rewards if r <= 0)
        logger.info(f"  traj_reward > 0: {pos}, <= 0: {neg}")

    # Check step_data structure - handle both dict and list types
    sd = sample.get("step_data", {})
    if isinstance(sd, dict):
        logger.info(f"  step_data type: dict, keys: {list(sd.keys())}")
        if "prompt" in sd and "completion" in sd:
            prompt_len = len(sd["prompt"]) if isinstance(sd["prompt"], list) else 1
            comp_len = len(sd["completion"]) if isinstance(sd["completion"], list) else 1
            logger.info(f"  prompt msgs: {prompt_len}, completion msgs: {comp_len}")
            # Estimate char length
            total_chars = sum(len(str(m.get("content", ""))) for m in sd["prompt"] + sd["completion"])
            logger.info(f"  ~{total_chars} total chars in first sample")
    elif isinstance(sd, list):
        logger.info(f"  step_data type: list (len {len(sd)})")
    else:
        logger.info(f"  step_data type: {type(sd).__name__}")


def clean_and_filter(dataset, num_proc=8):
    """
    Filter and clean dataset:
    1. Keep only positive-reward trajectories
    2. Extract step_data sub-structure
    3. Remove samples without valid prompt/completion
    """
    logger.info("Filtering traj_reward > 0...")
    ds = dataset.filter(
        lambda x: x.get("traj_reward", 0) > 0,
        num_proc=num_proc,
        desc="Filtering positive rewards",
    )
    logger.info(f"  After reward filter: {len(ds)} samples")

    # Extract step_data (handle both dict and list column types)
    logger.info("Extracting step_data...")

    def extract_step_data(x):
        sd = x["step_data"]
        if isinstance(sd, dict):
            return sd
        elif isinstance(sd, list) and len(sd) > 0:
            # If it's a list, take first element
            return sd[0]
        return sd

    ds = ds.map(
        extract_step_data,
        num_proc=num_proc,
        desc="Extracting step_data",
    )

    # Filter out samples missing prompt or completion
    logger.info("Filtering malformed samples...")
    ds = ds.filter(
        lambda x: (
            "prompt" in x
            and "completion" in x
            and x["prompt"]
            and x["completion"]
        ),
        num_proc=num_proc,
        desc="Filtering valid prompt/completion",
    )
    logger.info(f"  After cleaning: {len(ds)} samples")

    return ds


def flatten_conversation(sample):
    """Merge prompt and completion messages into a single conversation list."""
    prompt = sample.get("prompt", [])
    completion = sample.get("completion", [])
    return {"messages": prompt + completion}


def build_processed_dataset(dataset, tokenizer_id, num_proc=8):
    """
    Convert raw dataset into SFT-ready format:
    - Flattens prompt+completion into messages
    - Applies chat template to produce 'text' field
    """
    logger.info("Flattening prompt + completion into 'messages'...")
    ds = dataset.map(
        flatten_conversation,
        remove_columns=["prompt", "completion"],
        num_proc=num_proc,
        desc="Flattening conversations",
    )

    # Don't apply chat template here — do it in the training script
    # so tokenizer version matches the model exactly
    logger.info("Done. Chat template will be applied during training.")
    return ds


def main():
    parser = argparse.ArgumentParser(description="Preprocess Go-Browse dataset")
    parser.add_argument(
        "--local", type=str, default=None,
        help="Path to local dataset (e.g., data_local/go-browse-wa)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Process full dataset (default: first 50 for testing)"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.05,
        help="Validation split fraction (default: 0.05)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits"
    )
    parser.add_argument(
        "--num-proc", type=int, default=8,
        help="Number of processes for parallel map/filter"
    )
    parser.add_argument(
        "--output-dir", type=str, default="processed_data",
        help="Output directory for processed datasets"
    )
    args = parser.parse_args()

    # --- Load dataset ---
    if args.local:
        logger.info(f"Loading local dataset from: {args.local}")
        from datasets import Dataset
        ds = Dataset.load_from_disk(args.local)
        ds = ds.shuffle(seed=42)
        if not args.full:
            logger.info("Using first 50 samples for demo (use --full for all)")
            ds = ds.select(range(min(50, len(ds))))
    else:
        logger.info("Loading dataset from HuggingFace: apurvaga/go-browse-wa")
        from datasets import load_dataset
        if args.full:
            ds = load_dataset("apurvaga/go-browse-wa", split="train")
        else:
            logger.info("Loading first 50 samples for demo (use --full for all)")
            ds = load_dataset("apurvaga/go-browse-wa", split="train[:50]")

    inspect_dataset(ds, "Raw Dataset")

    # --- Clean and filter ---
    ds = clean_and_filter(ds, num_proc=args.num_proc)
    inspect_dataset(ds, "Cleaned Dataset")

    # --- Build processed format ---
    ds = build_processed_dataset(ds, tokenizer_id="placeholder", num_proc=args.num_proc)

    # --- Train / validation split ---
    logger.info(f"Creating train/val split (val_size={args.val_size})...")
    split_ds = ds.train_test_split(test_size=args.val_size, seed=args.seed)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    logger.info(f"  Train: {len(train_ds)} samples")
    logger.info(f"  Val:   {len(val_ds)} samples")

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train")
    val_path = os.path.join(args.output_dir, "val")

    logger.info(f"Saving train dataset to: {train_path}")
    train_ds.save_to_disk(train_path)

    logger.info(f"Saving val dataset to: {val_path}")
    val_ds.save_to_disk(val_path)

    # Save metadata
    meta = {
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "val_ratio": args.val_size,
        "seed": args.seed,
        "features": str(ds.features),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Print final example ---
    logger.info("--- Final Example (first train sample) ---")
    sample = train_ds[0]
    messages = sample["messages"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"][:300]
        logger.info(f"  [{role}] {content}...")

    logger.info("=== Preprocessing complete ===")
    logger.info(f"Output: {args.output_dir}/")
    logger.info(f"  train/  ({len(train_ds)} samples)")
    logger.info(f"  val/    ({len(val_ds)} samples)")


if __name__ == "__main__":
    main()
