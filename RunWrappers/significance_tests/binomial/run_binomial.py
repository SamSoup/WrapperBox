"""Runner for Obtaining and compiling model performance

Example usage: 

python3 main.py --args 

See python3 main.py --help
"""

from RunWrappers.significance_tests.binomial.binomial import (
    compute_binomial_p_value,
)
from utils.models import get_predictions
from datasets import concatenate_datasets, DatasetDict
from tqdm import tqdm
import pandas as pd
from utils.inference import compute_metrics
from utils.io import (
    load_dataset_from_hf,
    load_embeddings,
    load_labels_at_split,
    load_neural_predictions,
    load_wrapperbox,
    mkdir_if_not_exists,
)
import argparse
import numpy as np
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help="List of datasets to compute results for",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="List of neural models to compute results for",
    )
    parser.add_argument(
        "--wrappers",
        nargs="+",
        type=str,
        help=(
            "List of white box models to compute results for."
            "Must be one of KNN, SVM, DT, and LMeans"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pooler",
        type=str,
        default="mean_with_attention",
        help="Type of pooler",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=str,
        help="List of layer number, one per neural model",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        help=(
            "List of metrics to compute."
            "Must be one of [accuracy, f1, precision, recall]"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory",
    )
    args = parser.parse_args()

    # Print arguments
    print("Arguments passed:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


def load_embeddings_from_disk(
    dataset: str, model: str, seed: str, pooler: str, layer: int
):
    train_embeddings = load_embeddings(
        dataset=dataset,
        model=model,
        seed=seed,
        split="train",
        pooler=pooler,
        layer=layer,
    )

    eval_embeddings = load_embeddings(
        dataset=dataset,
        model=model,
        seed=seed,
        split="eval",
        pooler=pooler,
        layer=layer,
    )

    test_embeddings = load_embeddings(
        dataset=dataset,
        model=model,
        seed=seed,
        split="test",
        pooler=pooler,
        layer=layer,
    )

    train_eval_embeddings = np.vstack([train_embeddings, eval_embeddings])

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


def load_dataset_and_labels(dataset: str):
    dataset_dict = load_dataset_from_hf(dataset=dataset)
    train_labels = load_labels_at_split(dataset_dict, "train")
    eval_labels = load_labels_at_split(dataset_dict, "eval")
    test_labels = load_labels_at_split(dataset_dict, "test")

    # Combine train and eval
    train_eval_dataset = concatenate_datasets(
        [dataset_dict["train"], dataset_dict["eval"]]
    )

    train_eval_dataset_dict = DatasetDict(
        {"train": train_eval_dataset, "test": dataset_dict["test"]}
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


def create_result_df(models, metrics, classifiers):
    """
    Creates a empty dataframe with index = baseline + whitebox classifier names,
    and a multilevel column index of models * metrics
    """
    return pd.DataFrame(
        np.nan,
        index=classifiers,
        columns=pd.MultiIndex.from_product(
            [models, metrics], names=["models", "metrics"]
        ),
    )


if __name__ == "__main__":
    # Get arguments from command line
    args = get_args()

    # Run result computing for each dataset, each neural model/wrapper
    for dataset in tqdm(args.datasets, desc="Dataset"):
        results_df = create_result_df(
            models=args.models, metrics=args.metrics, classifiers=args.wrappers
        )
        (
            dataset_dict,
            train_eval_dataset_dict,
            train_labels,
            eval_labels,
            train_eval_labels,
            test_labels,
        ) = load_dataset_and_labels(dataset=dataset)
        is_multiclass = np.unique(test_labels).size > 2
        for model, layer in tqdm(
            zip(args.models, args.layers),
            desc="Neural Model",
            total=len(args.models),
        ):
            (
                train_embeddings,
                eval_embeddings,
                train_eval_embeddings,
                test_embeddings,
            ) = load_embeddings_from_disk(
                dataset=dataset,
                model=model,
                seed=args.seed,
                pooler=args.pooler,
                layer=layer,
            )
            # Load the original neural predictions instead
            og_predictions = load_neural_predictions(
                dataset=dataset, model=model, seed=42
            )
            for wrapper in args.wrappers:
                clf = load_wrapperbox(
                    dataset=dataset,
                    model=model,
                    seed=args.seed,
                    pooler=args.pooler,
                    wrapperbox=wrapper,
                )
                predictions = get_predictions(clf, test_embeddings)
                p_values = compute_binomial_p_value(
                    og_predictions,
                    predictions,
                    test_labels,
                    is_multiclass,
                )
                # Record metrics to dataframe
                for metric in args.metrics:
                    results_df.loc[wrapper][model][metric] = p_values[
                        f"test_{metric}"
                    ]

        # Load neural predictions, and subtract

        # Format and Output results
        results_df = results_df.round(3)

        print(results_df.to_string(index=True))
        print(results_df.to_latex())

        # Save results to disk
        mkdir_if_not_exists(args.output_dir)
        output_fname = os.path.join(
            args.output_dir, f"{dataset}_binomial_sigtest_results.csv"
        )
        results_df.to_csv(output_fname, index=True)
