"""Runner for Minimal Subsets

Example usage: 

python3 main.py --config conig/<X>

OR: pass in all other arguments
"""

import pandas as pd
from MinimalSubsetToFlipPredictions.Yang2023.recursive import IP_iterative
from MinimalSubsetToFlipPredictions.Yang2023.smallest_k import IP
from utils.constants.directory import (
    EMBEDDINGS_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    WORK_DIR,
)
from utils.io import load_pickle, mkdir_if_not_exists
import numpy as np
import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser()
    # Either give json, or provide all others
    parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--idx_start",
        type=int,
        default=0,
        help="Signify the 0-based index of the first test example to obtain the"
        " subsets for",
    )
    parser.add_argument(
        "--idx_end",
        type=int,
        default=None,
        help="Signify the 0-based index of the last test example to obtain the"
        " subsets for, must be greater than `start` if specified",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Name of the dataset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--thresh", type=float, default=0.5, help="Classification threshold"
    )
    parser.add_argument(
        "--l2",
        type=int,
        default=1,
        help="For yang's approach, set l2 penalty magnitude",
    )
    parser.add_argument(
        "--algorithm_type",
        type=str,
        default="fast",
        help="Algorithm 1 or Algorithm 2 as indicated by fast (1) or slow (2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="MinimalSubset",
        help="Output directory",
    )
    args = parser.parse_args()

    # If config argument is provided, load configuration from JSON file
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        # Overwrite args with config
        for key, value in config.items():
            setattr(args, key, value)

        # Check if range is valid
        if args.idx_end is not None:
            assert args.idx_end >= args.idx_start

    # Print arguments
    print("Arguments passed:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


def load_embeddings_from_disk(args: argparse.Namespace):
    dataset_name = args.dataset
    embeddings_path = os.path.join(EMBEDDINGS_DIR, dataset_name)

    train_embeddings = np.load(
        os.path.join(embeddings_path, "train.npy")
    ).squeeze()

    test_embeddings = np.load(
        os.path.join(embeddings_path, "test.npy")
    ).squeeze()

    test_embeddings = test_embeddings[args.idx_start : args.idx_end, :]

    # Print summary of embeddings
    print(f"Loaded train embeddings with {train_embeddings.shape} shape")
    print(f"Loaded test embeddings with {test_embeddings.shape} shape")

    return train_embeddings, test_embeddings


def load_labels(args: argparse.Namespace):
    dir = os.path.join(WORK_DIR, f"data/datasets/{args.dataset}")
    train_data = pd.read_csv(os.path.join(dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(dir, "test.csv"))

    train_labels = np.array(train_data["label"])
    test_labels = np.array(test_data["label"])

    return train_labels, test_labels


if __name__ == "__main__":
    # Get arguments from command line
    args = get_args()
    train_embeddings, test_embeddings = load_embeddings_from_disk(args=args)
    train_labels, test_labels = load_labels(args=args)

    # Check output dir is absolute path; if not, append RESULTS_DIR
    if not os.path.isabs(args.output_dir):
        args.output_dir = RESULTS_DIR / args.output_dir
    mkdir_if_not_exists(args.output_dir)

    # Load the logistic regression
    dataset_name = args.dataset
    model_path = SAVED_MODELS_DIR / dataset_name / "LogisticRegression.pkl"
    model = load_pickle(model_path)

    # Set up, before running
    X, y = {}, {}
    X["train"] = train_embeddings
    X["dev"] = test_embeddings
    y["train"] = train_labels
    y["dev"] = test_labels

    if args.algorithm_type == "fast":
        IP(
            model=model,
            dataname=dataset_name,
            X=X,
            y=y,
            l2=args.l2,
            thresh=args.thresh,
            output_dir=args.output_dir,
        )
    else:
        IP_iterative(
            model=model,
            dataname=dataset_name,
            X=X,
            y=y,
            l2=args.l2,
            thresh=args.thresh,
            output_dir=args.output_dir,
        )
