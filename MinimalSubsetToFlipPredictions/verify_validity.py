## This script is set up to verify the validity of subset indices
## and is meant to be run in parallel for different examples to check

import argparse
from fileinput import filename
import json
import os
import pickle
from typing import List

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sympy import false
from tqdm import tqdm

from utils.io import (
    load_dataset_from_hf,
    load_labels_at_split,
    load_embeddings,
    load_wrapperbox,
)
from datasets import concatenate_datasets, DatasetDict


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
    test_embeddings = test_embeddings[args.idx_start : args.idx_end, :]

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
    last = test_dataset.num_rows if args.idx_end is None else args.idx_end
    test_dataset = test_dataset.select(range(args.idx_start, last))
    test_labels = test_labels[args.idx_start : args.idx_end]
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


def retrain_and_evaluate_validity(
    clf: BaseEstimator,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    x_test: np.ndarray,
    indices_to_exclude: np.ndarray,
):
    train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
    train_mask[indices_to_exclude] = False
    reduced_embeddings = train_embeddings[train_mask]
    reduced_labels = train_labels[train_mask]
    old_pred = clf.predict(x_test.reshape(1, -1))[0]
    new_clf = clone(clf)
    new_clf.fit(reduced_embeddings, reduced_labels)
    new_pred = new_clf.predict(x_test.reshape(1, -1))[0]
    # this subset is valid only if new prediction does not equal old prediction
    return old_pred, new_pred, new_pred != old_pred


def evaluate_predictions(
    clf: BaseEstimator,
    flip_list: List[List[int]],
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    ex_indices_to_check: List[int],
):
    is_valid_subsets = []
    for test_ex_idx in tqdm(ex_indices_to_check):
        f_list = flip_list[test_ex_idx]
        # if flip list is empty: then it is obviously false
        if f_list is None or len(f_list) == 0:
            is_valid_subsets.append(False)
            continue
        _, _, is_valid_subset = retrain_and_evaluate_validity(
            clf=clf,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            x_test=test_embeddings[test_ex_idx],
            indices_to_exclude=f_list,
        )
        is_valid_subsets.append(is_valid_subset)

    return is_valid_subsets


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

    # Load pre-computed subsets
    fname = args.subsets_filename
    if fname is None:
        fname = f"{args.dataset}_{args.model}_{save_name}.pickle"
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

    total_valid = np.sum(is_subset_valid)
    print(f"Of the {total} checked subsets, only {total_valid} is valid")
    acc = total_valid / total * 100
    print(f"Validity of checked subsets: {acc:.2f}%")

    prefix = f"{args.dataset}_{args.model}_{save_name}_{args.idx_start}to{args.idx_end}"
    output_file_path = os.path.join(
        args.output_dir, f"{prefix}_is_valid_subsets.json"
    )
    with open(output_file_path, "w") as output_file:
        json.dump(is_subset_valid, output_file)
