"""Runner for Minimal Subsets

Example usage: 

python3 main.py --config conig/<X>

OR: pass in all other arguments
"""

from cgi import test
from MinimalSubsetToFlipPredictions.Yang2023.interface import (
    compute_minimal_subset_to_flip_predictions,
)
from MinimalSubsetToFlipPredictions.wrappers.factory import (
    FindMinimalSubsetFactory,
)
from utils.constants.directory import RESULTS_DIR
from utils.constants.models import (
    WRAPPER_BOXES_NEEDING_BATCHED_MINIMAL_SUBET_SEARCH,
)
from utils.io import mkdir_if_not_exists
from datasets import concatenate_datasets, DatasetDict
from data.embeddings.load_embeddings import load_saved_embeddings
from data.models import load_saved_wrapperbox_model
from data.datasets import load_dataset_from_hf, load_labels_at_split
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
        "--do_debug",
        type=bool,
        help="If true, truncate the test dataset to the"
        " the first `debug_examples` to test the script for completion",
    )
    parser.add_argument(
        "--debug_examples",
        type=int,
        help="Ignored unless do_debug is true",
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
        "--splits",
        type=int,
        default=10,
        help="How many batches to create for subset searching? Not used for KNN",
    )
    parser.add_argument(
        "--iterative_threshold",
        type=int,
        default=10,
        help="How many remaining examples to stop batch searching? Not used for KNN",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='RESULTS_DIR / "MinimalSubset"',
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

    return args


def load_embeddings(args: argparse.Namespace):
    train_embeddings = load_saved_embeddings(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        split="train",
        pooler=args.pooler,
        layer=args.layer,
    )

    eval_embeddings = load_saved_embeddings(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        split="eval",
        pooler=args.pooler,
        layer=args.layer,
    )

    test_embeddings = load_saved_embeddings(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        split="test",
        pooler=args.pooler,
        layer=args.layer,
    )

    train_eval_embeddings = np.vstack([train_embeddings, eval_embeddings])

    if args.do_debug:
        test_embeddings = test_embeddings[: args.debug_examples, :]

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
    if args.do_debug:
        test_dataset = test_dataset.select(range(args.debug_examples))
        test_labels = test_labels[: args.debug_examples]
        dataset_dict["test"] = test_dataset

    train_eval_dataset_dict = DatasetDict(
        {"train": train_eval_dataset, "test": test_dataset}
    )
    train_eval_labels = np.concatenate([train_labels, eval_labels])
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
    args = get_args()
    (
        train_embeddings,
        eval_embeddings,
        train_eval_embeddings,
        test_embeddings,
    ) = load_embeddings(args=args)
    (
        dataset_dict,
        train_eval_dataset_dict,
        train_labels,
        eval_labels,
        train_eval_labels,
        test_labels,
    ) = load_dataset_and_labels(args=args)

    if args.do_yang2023:
        # Running Yang et al
        compute_minimal_subset_to_flip_predictions(
            dataset_name=args.dataset,
            train_embeddings=train_embeddings,
            eval_embeddings=eval_embeddings,
            test_embeddings=test_embeddings,
            train_labels=train_labels,
            eval_labels=eval_labels,
            test_labels=test_labels,
            thresh=0.5,
            l2=500,
            output_dir="./results",
            algorithm=args.algorithm_type,
        )

    # Load handler
    factory = FindMinimalSubsetFactory()
    handler_class = factory.get_handler(
        f"{factory.interface_name}{args.wrapper_name}"
    )
    if args.wrapper_name in WRAPPER_BOXES_NEEDING_BATCHED_MINIMAL_SUBET_SEARCH:
        handler = handler_class(
            ITERATIVE_THRESHOLD=args.iterative_threshold, SPLITS=args.splits
        )
    else:
        handler = handler_class()

    # Load Wrapper box
    clf = load_saved_wrapperbox_model(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        pooler=args.pooler,
        wrapperbox=args.wrapper_name,
    )

    minimal_subset_indices = handler.find_minimal_subset(
        clf=clf,
        train_embeddings=train_eval_embeddings,
        test_embeddings=test_embeddings,
        train_labels=train_eval_labels,
    )

    # Check output dir is absolute path; if not, append RESULTS_DIR
    if not os.path.isabs(args.output_dir):
        args.output_dir = RESULTS_DIR / args.output_dir

    mkdir_if_not_exists(args.output_dir)

    handler.persist_to_disk(
        dataset=dataset_dict,
        dataset_name=args.dataset,
        model_name=args.model,
        wrapper_name=args.wrapper_name,
        minimal_subset_indices=minimal_subset_indices,
        output_dir=args.output_dir,
    )
