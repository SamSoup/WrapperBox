## This script is set up to verify the validity of subset indices
## and is meant to be run in parallel for different examples to check

from .evaluate import (
    compute_subset_metrics,
    evaluate_predictions,
)
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sympy import false
from tqdm import tqdm

from utils.constants.directory import RESULTS_DIR, WORK_DIR
from utils.io import (
    load_dataset_from_hf,
    load_labels_at_split,
    load_embeddings,
    load_wrapperbox,
    mkdir_if_not_exists,
)
from datasets import concatenate_datasets, DatasetDict
import argparse
import json
import os
import pickle
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # Either give json, or provide all others
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file, if any",
    )
    parser.add_argument(
        "--subsets_filename",
        type=str,
        default=None,
        help="Name of the file that contains the subset sizes.",
    )
    parser.add_argument(
        "--idx_start",
        type=int,
        default=0,
        help="Signify the 0-based index of the first test example to verify",
    )
    parser.add_argument(
        "--idx_end",
        type=int,
        default=None,
        help="Signify the 0-based index of the last test example to verify, "
        "must be greater than `start` if specified",
    )
    parser.add_argument(
        "--dataset", type=str, default="toxigen", help="Name of the dataset"
    )
    parser.add_argument(
        "--model", type=str, default="deberta-large", help="Name of the model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pooler",
        type=str,
        default="mean_with_attention",
        help="Type of pooler",
    )
    parser.add_argument("--layer", type=int, default=24, help="Layer number")
    parser.add_argument(
        "--do_yang2023",
        type=bool,
        default=False,
        help="If true, run minimal subset algorithm" " from yang et. al. 2023",
    )
    parser.add_argument(
        "--algorithm_type",
        type=str,
        default="fast",
        help="Algorithm 1 or Algorithm 2 as indicated by fast (1) or slow (2)",
    )
    parser.add_argument(
        "--wrapper_name",
        type=str,
        default="KNN",
        help="Name of the wrapper. Will be ignored if do_yang2023",
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
    train_embeddings = load_embeddings(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        split="train",
        pooler=args.pooler,
        layer=args.layer,
    )

    eval_embeddings = load_embeddings(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        split="eval",
        pooler=args.pooler,
        layer=args.layer,
    )

    test_embeddings = load_embeddings(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        split="test",
        pooler=args.pooler,
        layer=args.layer,
    )

    train_eval_embeddings = np.vstack([train_embeddings, eval_embeddings])
    # test_embeddings = test_embeddings[args.idx_start : args.idx_end, :]
    # Print summary of embeddings
    print(f"Loaded train embeddings with {train_embeddings.shape} shape")
    print(f"Loaded eval embeddings with {eval_embeddings.shape} shape")
    print(f"Loaded test embeddings with {test_embeddings.shape} shape")

    return (
        train_embeddings,
        eval_embeddings,
        train_eval_embeddings,
        test_embeddings,
    )


def load_dataset_and_labels(args: argparse.Namespace):
    dataset_dict = load_dataset_from_hf(dataset=args.dataset)
    train_labels = load_labels_at_split(dataset_dict, "train")
    eval_labels = load_labels_at_split(dataset_dict, "eval")
    test_labels = load_labels_at_split(dataset_dict, "test")

    # Combine train and eval
    train_eval_dataset = concatenate_datasets(
        [dataset_dict["train"], dataset_dict["eval"]]
    )
    test_dataset = dataset_dict["test"]
    # last = test_dataset.num_rows if args.idx_end is None else args.idx_end
    # last = min(test_dataset.num_rows, args.idx_end)
    # test_dataset = test_dataset.select(range(args.idx_start, last))
    # test_labels = test_labels[args.idx_start : args.idx_end]
    dataset_dict["test"] = test_dataset

    train_eval_dataset_dict = DatasetDict(
        {"train": train_eval_dataset, "test": test_dataset}
    )
    train_eval_labels = np.concatenate([train_labels, eval_labels])

    # Print summary of datasets
    for name, dataset in dataset_dict.items():
        print(f"Loaded {name} dataset with {dataset.num_rows} rows")

    return (
        dataset_dict,
        train_eval_dataset_dict,
        train_labels,
        eval_labels,
        train_eval_labels,
        test_labels,
    )


if __name__ == "__main__":
    # Get arguments from command line
    args = parse_args()
    (
        train_embeddings,
        eval_embeddings,
        train_eval_embeddings,
        test_embeddings,
    ) = load_embeddings_from_disk(args=args)
    (
        dataset_dict,
        train_eval_dataset_dict,
        train_labels,
        eval_labels,
        train_eval_labels,
        test_labels,
    ) = load_dataset_and_labels(args=args)

    # Load pre-trained classifiers
    if args.do_yang2023:
        clf = load_wrapperbox(
            dataset=args.dataset,
            model=args.model,
            seed=args.seed,
            pooler=args.pooler,
            wrapperbox="LogisticRegression",
        )
        save_name = f"yang_{args.algorithm_type}"
    else:
        clf = load_wrapperbox(
            dataset=args.dataset,
            model=args.model,
            seed=args.seed,
            pooler=args.pooler,
            wrapperbox=args.wrapper_name,
        )
        save_name = args.wrapper_name

    # Load pre-computed subsets from directory
    fname = args.subsets_filename
    if fname is None:
        fname = f"{args.dataset}_{args.model}_{save_name}.pickle"
    if not os.path.isabs(fname):
        fname = os.path.join(WORK_DIR, fname)
    with open(fname, "rb") as handle:
        flip_list = pickle.load(handle)

    # Finally: evaluate and retrain
    ex_indices_to_check = np.arange(args.idx_start, args.idx_end)
    total = ex_indices_to_check.size
    print(
        f"Checking subsets for test examples {args.idx_start} to {args.idx_end}."
        f"\nTotal: {total} examples."
    )
    is_subset_valid = evaluate_predictions(
        clf=clf,
        flip_list=flip_list,
        train_embeddings=train_eval_embeddings,
        train_labels=train_eval_labels,
        test_embeddings=test_embeddings,
        ex_indices_to_check=ex_indices_to_check,
    )

    # Check output dir is absolute path; if not, append RESULTS_DIR
    if not os.path.isabs(args.output_dir):
        args.output_dir = RESULTS_DIR / args.output_dir
    mkdir_if_not_exists(args.output_dir)

    # Save is subset to disk
    prefix = f"{args.dataset}_{args.model}_{save_name}_{args.idx_start}to{args.idx_end}"
    fname = f"{prefix}_is_valid_subsets.pickle"
    output_file_path = os.path.join(args.output_dir, fname)
    with open(output_file_path, "wb") as f:
        pickle.dump(is_subset_valid, f)
