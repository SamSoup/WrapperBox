"""
Command line runner file to compute representations.

Usage: python3 main.py --help
"""

import argparse
import json
import os
import numpy as np
from datasets import load_dataset
from ComputeRepresentations.ModelForSentenceLevelRepresentation import (
    ModelForSentenceLevelRepresentation,
)
from utils.constants.directory import CACHE_DIR
from utils.io import mkdir_if_not_exists


def get_args():
    parser = argparse.ArgumentParser()
    # Either give json, or provide all others
    parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Specify a path or a hub location to a Dataset",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Specify a path or a hub location to a HF model.",
    )
    parser.add_argument(
        "--load_half_precision",
        type=bool,
        help="Should we load the model in half precision to save mem?",
    )
    parser.add_argument(
        "--pooler",
        type=str,
        help="Name of pooler functions to sue. See `EmbeddingPooler` for options.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Max length of input tokens.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for resultant representations.",
    )
    args = parser.parse_args()

    # If config argument is provided, load configuration from JSON file
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        # Overwrite args with config
        for key, value in config.items():
            setattr(args, key, value)

    print("Arguments passed:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


if __name__ == "__main__":
    args = get_args()
    mkdir_if_not_exists(args.output_dir)
    datasets = load_dataset(
        args.dataset_name_or_path, use_auth_token=True, cache_dir=CACHE_DIR
    )
    model = ModelForSentenceLevelRepresentation(
        args.model_name_or_path, args.pooler, args.load_half_precision
    )
    for split, dataset in datasets.items():
        print(f"***** Computing Representations for {split} dataset *****")
        # Assume that the column to compute representation is for is 'text'
        representations = model.extract_representations(
            texts=dataset["text"],
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        save_path = os.path.join(args.output_dir, f"{split}.npy")
        # Save representations as numpy file
        np.save(save_path, representations.numpy())
